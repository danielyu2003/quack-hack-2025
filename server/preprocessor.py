import re

class Preprocessor:
    """
    Class encapsulating logic for text preprocessing.
    """
    def __init__(self):
        self.isNotAlpha = re.compile(r"[^a-z\s]+")  # Remove non-alphabetic characters
        self.isUrl = re.compile(r"(https?://|www\.)\S+")  # Remove URLs
        self.isStopword = re.compile(r"\b(i|me|my|...|now)\b")  # Remove common stopwords
        self.isSuffix = re.compile(r"\B(ings?|e[sdr]|st?|ly|ment|ness|ion)\b")  # Remove common suffixes for stemming
        self.afterExcept = re.compile(r"\bexcept\b.*") # Remove anything after and including the word 'except'

    def __call__(self, x):
        return self.pipeline(x)

    def pipeline(self, text):
        """
        Apply text cleaning steps sequentially.

        Args:
            text (str): Raw text input.

        Returns:
            list of str: Processed word tokens.
        """
        line = text.lower() # first ensure that all text is lowercase
        line = self.isUrl.sub('', line)
        line = self.isNotAlpha.sub('', line)
        line = self.isStopword.sub('', line)
        line = self.isSuffix.sub('', line)
        line = self.afterExcept.sub('', line)
        return line
