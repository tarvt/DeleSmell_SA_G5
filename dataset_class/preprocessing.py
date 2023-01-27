from string import punctuation
import re
class preprocessing():

    def lm_find_unchinese(self, line):
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        unchinese = re.sub(pattern, "", line)  # Exclude Chinese characters
        unchinese = re.sub('[{}]'.format(punctuation), "", unchinese)  # Exclude Chinese characters

        return unchinese

    def del_stopwords(self, line):
        dicts = {'\n': '', '!': '', '"': '', '：': '', '#': '', '$': '', '%': '', '&': '', "'": '', '(': '', ')': '',
                 '*': '', '+': '', ',': '', '-': '', '.': '', '/': '', ':': '', ';': '', '<': '', '=': '', '>': '',
                 '?': '', '@': '', '[': '', '\\': '', ']': '', '^': '', '_': '', '`': '', '{': '', '|': '', '}': '',
                 '~': ''}
        punc_table = str.maketrans(dicts)
        new_line = line.translate(punc_table)

        return new_line