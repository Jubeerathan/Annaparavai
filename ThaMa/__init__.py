"""
AI Human Text Detector
A package for detecting AI-generated and human-written text in Tamil and Malayalam.
"""

__version__ = "0.1.0"
__author__ = "Jubeerathan Thevakumar, Luheerathan Thevakumar"
__email__ = "jubeerathan.20@cse.mrt.ac.lk, the.luheerathan@gmail.com"

from .model import TextDetector

SUPPORTED_LANGUAGES = TextDetector.SUPPORTED_LANGUAGES

__all__ = ['TextDetector', 'SUPPORTED_LANGUAGES']
