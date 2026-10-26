"""Offline regression tests. Run: python -m unittest discover -s tests -v."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'scripts'))
import bona_movie_production as bona


class TaskPollingTests(unittest.TestCase):
    def setUp(self):
        self.client = bona.BonaClient('https://example.invalid', 'https://example.invalid/login',
                                      None, 'test-token', 10)

    def test_processing_transitions_to_terminal_status(self):
        for processing in (1, '1'):
            for terminal in (9, '9', -1, '-1'):
                for nested in (False, True):
                    with self.subTest(processing=processing, terminal=terminal, nested=nested):
                        def response(status):
                            data = {'status': status, 'result_url': '',
                                    'error_code': 'GEN_FAILED', 'error_message': 'generation failed'}
                            return {'data': data} if nested else data
                        final = response(terminal)
                        with patch.object(self.client, 'query_task', side_effect=[response(processing), final]) as query, \
                                patch.object(bona.time, 'time', return_value=0), \
                                patch.object(bona.time, 'sleep') as sleep:
                            result = self.client.wait_task('task', 'thread', 'model', 5, 30)
                        self.assertIs(result, final)
                        self.assertEqual(query.call_args_list, [call(task_id='task', thread_id='thread', model_name='model')] * 2)
                        sleep.assert_called_once_with(5)

    def test_terminal_status_returns_immediately(self):
        for status in (9, '9', -1, '-1', 'success', 'failed'):
            with self.subTest(status=status):
                final = {'data': {'status': status, 'error_code': 'E', 'error_message': 'details'}}
                with patch.object(self.client, 'query_task', return_value=final) as query, \
                        patch.object(bona.time, 'time', return_value=0), \
                        patch.object(bona.time, 'sleep') as sleep:
                    self.assertIs(self.client.wait_task('task', None, None, 5, 30), final)
                query.assert_called_once()
                sleep.assert_not_called()

    def test_nonterminal_status_times_out(self):
        for status in (1, '1', 0, '0', 'processing', 'unknown', None):
            with self.subTest(status=status):
                with patch.object(self.client, 'query_task', return_value={'data': {'status': status}}) as query, \
                        patch.object(bona.time, 'time', side_effect=[0, 0, 5, 11]), \
                        patch.object(bona.time, 'sleep') as sleep:
                    with self.assertRaisesRegex(TimeoutError, 'task task did not finish within 10 seconds'):
                        self.client.wait_task('task', None, None, 5, 10)
                self.assertEqual(query.call_count, 2)
                self.assertEqual(sleep.call_args_list, [call(5), call(5)])

    def test_image_and_video_wait_cli(self):
        for kind in ('image', 'video'):
            for processing in (1, '1'):
                for terminal in (9, '9', -1, '-1'):
                    with self.subTest(kind=kind, processing=processing, terminal=terminal):
                        final = {'data': {'status': terminal, 'result_url': '',
                                          'error_code': 'E_GENERATE', 'error_message': 'failure details'}}
                        replies = [{'data': {'status': processing, 'result_url': ''}}, final]
                        http = [Mock(status_code=200, json=Mock(return_value=r)) for r in replies]
                        created = {'data': {'task_id': 'task'}}
                        output = io.StringIO()
                        argv = ['bona', 'create-' + kind, '--prompt', 'A cinematic scene',
                                '--access-token', 'test-token', '--thread-id', 'thread', '--wait']
                        with patch.object(sys, 'argv', argv), \
                                patch.object(bona.requests, 'post', return_value=Mock(status_code=200, json=Mock(return_value=created))) as post, \
                                patch.object(bona.requests, 'get', side_effect=http) as get, \
                                patch.object(bona.time, 'time', return_value=0), \
                                patch.object(bona.time, 'sleep') as sleep, \
                                contextlib.redirect_stdout(output):
                            self.assertEqual(bona.main(), 0)
                        self.assertEqual(json.loads(output.getvalue())['query_response'], final)
                        self.assertTrue(post.call_args.args[0].endswith('/' + kind))
                        self.assertEqual(get.call_count, 2)
                        self.assertEqual(get.call_args.kwargs['params']['thread_id'], 'thread')
                        self.assertIn('model_name', get.call_args.kwargs['params'])
                        sleep.assert_called_once_with(5)

    def test_image_and_video_wait_cli_timeout(self):
        for kind in ('image', 'video'):
            with self.subTest(kind=kind):
                output, errors = io.StringIO(), io.StringIO()
                argv = ['bona', 'create-' + kind, '--prompt', 'A cinematic scene',
                        '--access-token', 'test-token', '--wait', '--max-wait', '10']
                with patch.object(sys, 'argv', argv), \
                        patch.object(bona.requests, 'post', return_value=Mock(status_code=200, json=Mock(return_value={'task_id': 'task'}))), \
                        patch.object(bona.requests, 'get', return_value=Mock(status_code=200, json=Mock(return_value={'status': 1}))), \
                        patch.object(bona.time, 'time', side_effect=[0, 0, 5, 11]), \
                        patch.object(bona.time, 'sleep'), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    self.assertEqual(bona.main(), 1)
                self.assertEqual(output.getvalue(), '')
                self.assertIn('did not finish within 10 seconds', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
