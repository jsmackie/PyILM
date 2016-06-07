#!/usr/bin/env python
#-*- coding: utf-8 -*-

import collections
from errors import ModelError

class Feature(object):
    """
    Attributes
    name: str , e.g 'voice'
    sign: str, e.g. '+' or '-'.
    """
    __slots__=['sign','name']

    def __init__(self, sign, name):
        self.sign = sign
        self.name = name


    def __getitem__(self, key):
        return self.__str__()[key]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ''.join([self.sign,self.name])

    def __eq__(self, other):
        """
        Two features compare equal if they have both the same name and same sign
        """
        return self.__str__() == other.__str__()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self,other):
        """
        If the features are the same, then sort '-' before '+'
        Otherwise, sort alphabetically
        """

        if self.name == other.name:

            if self.sign == '+' and other.sign == '+':
                return False
            if self.sign =='+' and other.sign == '-':
                return False
            if self.sign == '-' and other.sign == '-':
                return False
            if self.sign =='-' and other.sign == '+':
                return True
            if self.sign == '.' and other.sign == 'n':
                return True
            if self.sign == 'n' and other.sign == '.':
                return False
            if (self.sign == '+' or self.sign=='-') and (other.sign == 'n' or other.sign == '.'):
                return False

        else:
            return self.name[0] < other.name[0]

    def __le__(self,other):

        if self.name == other.name:
            if self.sign == '+' and other.sign == '+':
                return True
            if self.sign =='+' and other.sign == '-':
                return False
            if self.sign == '-' and other.sign == '-':
                return True
            if self.sign =='-' and other.sign == '+':
                return True
            if self.sign == '.' and other.sign == 'n':
                return True
            if self.sign == 'n' and other.sign == '.':
                return False
            if (self.sign == '+' or self.sign=='-') and (other.sign == 'n' or other.sign == '.'):
                return False

        else:
            return self.name[0] <= other.name[0]


    def __gt__(self,other):
        return not self.__le__(other)

    def __ge__(self,other):
        return not self.__lt__(other)

    def __hash__(self):
        return hash(tuple([self.__str__()]))

    def isplus(self):
        if self.sign == '+':
            return True
        else:
            return False

    def isminus(self):
        return not self.isplus()


class Environment(object):

    def __init__(self,lhs,rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __hash__(self):
        return id(self.lhs)+id(self.rhs)

    def __eq__(self,other):

        l_match = False
        r_match = False

        if not self.lhs or not other.lhs:
            #no left hand side specified, automatic match
            l_match = True
        elif self.lhs < other.lhs:
            l_match = True

        if not self.rhs or not other.rhs:
            #no right hand side specified, automatic match
            r_match = True
        elif self.rhs < other.rhs:
            r_match = True

        return l_match and r_match

    def __ne__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '_'.join([str(self.lhs), str(self.rhs)])

    def __len__(self):
        return 2

    def __contains__(self,element):
        try:
            match_symbol = element.symbol
        except AttributeError:
            match_symbol = element

        if not match_symbol:
            return True #empty symbol is always in the environment

        if (self.lhs.symbol == match_symbol or self.rhs.symbol == match_symbol):
            return True
        else:
            return False


class PrintableOrderedDictionary(collections.OrderedDict):

    def __repr__(self):
        output = list()
        for k,v in iter(self.items()):
            output.append(str(v))
        return ','.join(output)

    def __str__(self):
        return self.__repr__()


class PrintableDictionary(dict):

    def __repr__(self):
        output = list()
        for k,v in iter(self.items()):
            output.append(str(v))
        return ','.join(output)

    def __str__(self):
        return self.__repr__()

class Lexicon(dict):

    def get_full_lexicon(self, exclude=None):
        """
        Return all the words in the lexicon as a list.
        """

        full_lex = list()
        if exclude is None:
            exclude = list()

        keys = (k for k in self.keys() if not k in exclude)
        for k in keys:
            full_lex.extend(self[k])

        return full_lex

    def iter_full_lexicon(self, exclude=None):
        """
        Return all the words in the lexicon as a generator.

        """
        if exclude is None:
            exclude = list()

        keys = (k for k in self.keys() if not k in exclude)
        for k in keys:
            yield self[k]


    def __repr__(self):
        output = list()
        for meaning, wordlist in iter(self.items()):
            meaning_str = str(meaning)+': '
            word_list = list()
            for word in wordlist:
                word_str = u''.join([seg.symbol for seg in word.string])
                word_str = word_str + '(' + str(word.freq) + ') '
                word_list.append(word_str)
            output.append(meaning_str+u','.join(word_list))
        output = u'\n'.join(output)
        return output

    def __str__(self):
        return self.__repr__()


class Segment(object):

    __slots__=['symbol','features','envs','freq','distributions']
    def __init__(self, symbol, features, envs=None):

        self.symbol = symbol
        self.features = PrintableOrderedDictionary()
        #inherits from dict, just has a nicer __str__()
        self.init_features(features)
        self.envs = dict()
        self.freq = 0
        self.distributions = dict()

        if envs is None:
            pass# do nothing right now

        else:
            for e in envs:
                self.envs[e] += 1


    def init_features(self, features):

        if isinstance(features, list):
            for f in features:
                if f == '#':
                    self.features['#'] = Feature('','#')
                elif isinstance(f,Feature):
                    self.features[f.name] = f
                elif isinstance(f,str):
                    f = Feature(f[0], f[1:])
                    self.features[f.name] = f
                else:
                    raise ModelError('Features cannot be of type {}'.format(type(f)))

        elif isinstance(features,dict):
            self.features = features

        return

    @property
    def feature_list(self):
        return [str(value) for value in self.features.values()]

    def add_feature(self,f):

        if isinstance(f,Feature):
            self.features[f.name] = str(f)

        elif f is None or f == '#':
            self.features['#'] = Feature('','#')
            #This is a special case for the word boundary symbol

        else:
            f = Feature(f[0], f[1:])
            #split apart the sign and the name
            self.features[f.name] = f

        return

    def update_envs(self,e):

        if e in self.envs:
            self.envs[e] += 1
        else:
            self.envs[e] = 1

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.symbol

    def __len__(self):
        return 1

    def __eq__(self, other):
        """
        Two segments compare equal if all of their feature values match
        the symbols assigned to them are not taken into consideration
        context is also not important - segments with the same features but
        which appear in complementary distribution, or overlapping distribution,
        are treated as equal
        """

        for feature in self.features:
            if isinstance(other, str):
                if self.features[feature] != other:
                    return False
            elif self.features[feature] != other.features[feature]:
                return False
        return True

    def __lt__(self,other):
        """
        SegmentA < SegmentB if all the features in A are in B.
        This is mainly useful for comparing environments, which may not be
        fully specified for all features
        """
        if not len(self):
            return True #EmptySegment
        elif self.symbol == '#' and other.symbol == '#':
            return True
        elif self.symbol == '#' and not other.symbol == '#':
            return False
        elif not self.symbol == '#' and other.symbol == '#':
            return False
        else:
            return all([self.features[f] == other.features[f] for f in self.features])

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_vowel(self):
        return self.features['voc'] == '+voc'

class EmptySegment(Segment):

    def __len__(self):
        return 0


class Sound(Segment):

    __slots__=['symbol','features']
    def __init__(self,symbol,features):
        self.symbol = symbol
        self.features = PrintableOrderedDictionary()
        for feature in features:
            self.features[feature.name] = feature

    def reload_symbols(self):
        """
        Make sure all the tokens bear the correct label, generally useful
        during the learning process
        """
        for feature in self.features:
            self.features[feature].label = self.symbol


class Token(object):
    """
    Represents the pronunciation of a Feature
    """
    __slots__=['name','value','label','env','entry_time']
    time = 0

    @classmethod
    def stamp(cls):
        return Token.time

    @classmethod
    def increase_timer(cls):
        cls.time += 1
        return

    def __init__(self,name,value,label,env,time=None):
        self.name = name #name of a phonetic feature
        self.value = value #value of phonetic feature
        self.label = label #the name of the segment of which this token is an exemplar
        self.env = env
        self.entry_time = time

    def __str__(self):
        return ''.join([self.name,'->',str(self.value),' (',self.label,')'])

    def __repr__(self):
        return self.__str__()

    def __lt__(self,other):
        return self.value < other.value

    def __le__(self,other):
        return self.value <= other.value

    def __gt__(self,other):
        return self.value > other.value

    def __ge__(self,other):
        return self.value >= other.value

    def __eq__(self,other):
        if not isinstance(other, Token):
            raise ModelError('cannot compare Tokens with %s'%type(other))
        else:
            return self.value == other.value

class FeatureSpace(dict):


    def __str__(self):
        output = list()
        for k in self.keys():
            output.append(' '.join([l.label for l in self[k]]))
        output = '\n'.join(output)
        return output

    def __repr__(self):
        return self.__str__()


class Word(object):

    __slots__=['string','meaning','freq']
    meaning_counter = 0

    @classmethod
    def new_meaning(cls):
        cls.meaning_counter += 1
        return Word.meaning_counter

    def __init__(self,string,meaning=None):
        self.string = string
        if meaning is None:
            self.meaning = Word.new_meaning()
        else:
            self.meaning = meaning
        self.freq = 1

    def update_string(self, pos, seg):
        self.string[pos] = seg

    def __hash__(self):
        return sum(id(s) for s in self.string)+self.meaning

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ''.join([s.symbol for s in self.string])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.meaning == other.meaning) and (self.string == other.string)

    def __contains__(self,item):
        return item in self.string

    def __len__(self):
        return len(self.string)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError('index must be an integer')
        else:
            return self.string[key]

    def __setitem__(self,key,value):
        if not isinstance(key,int):
            raise TypeError('index must be an integer')
        self.string[key] = value