import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_text

def test_clean_text_basic():
    result = clean_text("This is FAKE news!!!")
    assert isinstance(result, str)
    assert result == result.lower()

def test_clean_text_removes_urls():
    result = clean_text("Visit http://fake.com for news")
    assert "http" not in result
    assert "fake.com" not in result

def test_clean_text_handles_empty():
    result = clean_text("")
    assert result == ""

def test_clean_text_handles_none():
    result = clean_text(None)
    assert result == ""

def test_clean_text_removes_numbers():
    result = clean_text("There were 1000 people at the rally")
    assert "1000" not in result