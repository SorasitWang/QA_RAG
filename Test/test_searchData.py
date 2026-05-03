import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch, mock_open

from Controller.SearchData import cleanText, fetchDataFromFolder


class TestFetchDataFromFolder(unittest.TestCase):

    @patch('os.listdir', return_value=['a.txt', 'b.txt'])
    @patch('builtins.open', mock_open(read_data="file content"))
    def test_returns_one_source_per_file(self, mock_listdir):
        result = fetchDataFromFolder('/folder')
        self.assertEqual(len(result), 2)

    @patch('os.listdir', return_value=['doc.txt'])
    @patch('builtins.open', mock_open(read_data="hello world"))
    def test_source_content_matches_file_contents(self, mock_listdir):
        result = fetchDataFromFolder('/folder')
        self.assertEqual(result[0].content, "hello world")

    @patch('os.listdir', return_value=['doc.txt'])
    @patch('builtins.open', mock_open(read_data="data"))
    def test_source_type_is_text(self, mock_listdir):
        result = fetchDataFromFolder('/folder')
        self.assertEqual(result[0].type, 'text')

    @patch('os.listdir', return_value=['doc.txt'])
    @patch('builtins.open', mock_open(read_data="data"))
    def test_source_tags_are_empty(self, mock_listdir):
        result = fetchDataFromFolder('/folder')
        self.assertEqual(result[0].tags, [])

    @patch('os.listdir', return_value=[])
    def test_empty_folder_returns_empty_list(self, mock_listdir):
        result = fetchDataFromFolder('/folder')
        self.assertEqual(result, [])


class TestCleanText(unittest.TestCase):

    def test_keeps_normal_lines(self):
        text = "This is a normal sentence that should be kept."
        self.assertEqual(cleanText(text), text)

    def test_removes_short_lines(self):
        text = "OK\nThis line is long enough to be kept."
        result = cleanText(text)
        self.assertNotIn("OK", result)
        self.assertIn("This line is long enough to be kept.", result)

    def test_removes_url(self):
        text = "Visit https://example.com/promo for details.\nThis is a real sentence with enough length."
        result = cleanText(text)
        self.assertNotIn("https://example.com", result)
        self.assertIn("This is a real sentence", result)

    def test_removes_www_url(self):
        text = "www.advertiser.com/signup\nThis is a real sentence with enough length."
        result = cleanText(text)
        self.assertNotIn("www.advertiser.com", result)

    def test_removes_email(self):
        text = "Contact us at support@example.com today\nThis is a real sentence with enough length."
        result = cleanText(text)
        self.assertNotIn("support@example.com", result)

    def test_removes_phone_number(self):
        text = "Call +1 (800) 555-1234 now\nThis is a real sentence with enough length."
        result = cleanText(text)
        self.assertNotIn("+1 (800) 555-1234", result)

    def test_empty_string_returns_empty(self):
        self.assertEqual(cleanText(""), "")

    def test_all_noisy_lines_returns_empty(self):
        text = "OK\nhttps://ad.example.com\nfoo@bar.com"
        self.assertEqual(cleanText(text), "")

    def test_strips_leading_trailing_whitespace(self):
        text = "   This is a normal sentence that should be kept.   "
        result = cleanText(text)
        self.assertEqual(result, "This is a normal sentence that should be kept.")


if __name__ == '__main__':
    unittest.main()
