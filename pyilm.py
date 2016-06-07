#!/usr/bin/env python
#-*- coding: utf-8 -*-

#standard library imports
import random
import os
from codecs import open
import itertools
import shutil
import math
import sys
import configparser
import time

#Numeric computing imports
from numpy import array
import scipy
import scipy.stats
import scipy.cluster.vq as Cluster

#PyILM-specific imports
from change import Misperception
from segbase import Segbase
from phonology import *

#Errors
from numpy.linalg import LinAlgError
from errors import ModelError

#global constant
E = math.e


class Simulation(object):
    """
    Container object for the entire simulation.
    """
    #all of these attributes can be over-ridden using a config.ini file
    basedir = os.path.join(os.getcwd(), 'Simulation Results')
    name = None
    auto_increase_lexicon_size = False
    generations = 30
    initial_inventory = 15,5
    initial_lexicon_size = 25
    min_word_length = 1
    max_word_length = 2
    max_inventions = 5
    invention_rate = 0.0
    make_minimal_pair = 0.5
    phonotactics = 'CVC'
    features_file = 'ipa2spe.txt'
    misperception_set = list()
    learning_algorithm = 'exemplars'
    segbase = None
    minimum_activation_level = 0.7
    minimum_category_distance = 0.4
    inherent_variation = 0
    allow_unmarked = False
    seed = random.randint(1,10000)
    filename = 'config.ini'
    cycles = 1
    samplesize = .7
    density_type = 'gaussian'
    medianormean = 'median'
    minimum_repetitions = 1
    save_sim = False
    min_spread = 0.05
    max_lexicon_size = 50
    minimize_features = False
    feature_division = 'absolute'
    seg_specific_misperceptions = False
    original_misperceptions = list()
    initial_words = list()

    #this list is used by visualizer.py
    _user_access = ['generations','initial_lexicon_size', 'initial_inventory',
    'min_word_length','max_word_length', 'max_inventions', 'invention_rate','phonotactics',
    'features_file','seed','minimum_activation_level', 'samplesize',
    'max_lexicon_size', 'minimum_repetitions', 'feature_division', 'features_file']


    def __init__(self, config_file=None, logdir=None):

        print('initializing simulation...')

        self.config_parse(config_file)
        self.config_file = config_file
        self.current_generation = 0

        if not os.path.exists(Simulation.basedir):
            os.mkdir(Simulation.basedir)

        if not os.path.exists(os.path.join(os.getcwd(), 'config')):
            os.mkdir(os.path.join(os.getcwd(), 'config'))

        if logdir is not None:#Temporary hack to deal with missing simulations
            self.logdir = logdir

        elif Simulation.name is None: #This is the typical case
            dircreated = False
            n = 0
            while not dircreated:
                try:
                    dirname = os.path.join(Simulation.basedir, 'Simulation output ({})'.format(n))
                    os.mkdir(dirname)
                    self.logdir = dirname
                    dircreated = True

                except OSError:  # folder exists already
                    n += 1

        else:
            self.logdir = os.path.join(Simulation.basedir, Simulation.name)
            if not os.path.exists(self.logdir):
                os.mkdir(self.logdir)
            else:
                raise ModelError(('The name given to the Simulation matches a name that already exists in your results '
                                  'directory. Please change the name of the simulation (or leave that option blank)'))

        #Initialize various aspects of the Simulation
        self.init_segbase_and_inventory()
        self.init_max_phonotactics()
        if Simulation.seg_specific_misperceptions:
            Simulation.original_misperceptions = [m for m in Simulation.misperception_set]
            self.generate_seg_specific_misperceptions()

        #Make a record the initial state of the Simulation
        if isinstance(self.initial_inventory, tuple):
            with open(os.path.join(self.logdir, 'initial_inventory.txt'), 'w', encoding='utf-8') as f:
                print(','.join([str(seg) for seg in Simulation.initial_inventory_list]), file=f)

        with open(os.path.join(self.logdir, 'misperceptions.txt'), mode='w', encoding='utf-8') as f:
            for misp in Simulation.misperception_set:
                print(misp.fileprint(), file=f)

        seed_name = ''.join(['RANDOM SEED ', str(Simulation.seed)])

        f = open(os.path.join(self.logdir, seed_name), mode='w', encoding='utf-8')
        f.close()  #intentionally empty file

        shutil.copy2(os.path.join(os.getcwd(), self.features_file), self.logdir)
        try:
            if self.config_file:
                shutil.copy2(os.path.join(os.getcwd(), 'config', self.config_file), self.logdir)
        except shutil.SameFileError:
            pass #Occurs when loading an existing simulation with an existing config file

    def generate_seg_specific_misperceptions(self):
        new_misperceptions = list()
        for seg in Simulation.initial_inventory_list:
            if seg.symbol == '#':
                continue
            used = list()
            for misperception in Simulation.original_misperceptions:
                #check which of the original class-level misperceptions would apply at this generation
                #and make new segment-specific misperceptions instead, each of which applies to a new
                #feature dimension (randomly selected)
                if all(feature in seg.feature_list for feature in misperception.filter):
                    new_filter = seg.feature_list
                    if len(used) == len(seg.features):
                        used = list()

                    new_target = random.choice([f for f in seg.features if not f in used])
                    used.append(new_target)
                    new_name = '--'.join([seg.symbol, misperception.name])
                    #randomly flip the direction of the misperception
                    if random.random() > 0.5:
                        misperception.salience *= -1

                    new_misp = Misperception(new_name,
                                             new_filter,
                                             new_target,
                                             misperception.salience,
                                             misperception.env,
                                             misperception.p)
                    new_misperceptions.append(new_misp)
        Simulation.misperception_set = new_misperceptions

    @classmethod
    def get_environment(cls, pos, word):

        if pos == 0:
            #Word-initial position, there's no segment object on the left
            #lhs = '#_'
            lhs = Segment('#', ['#'], None)
        else:
            #Word-medial position, get the segment object on the left
            lhs = word[pos-1]

        if pos == len(word)-1:
            #Word-final position, there's no segment object to the right
            #rhs = '_#'
            rhs = Segment('#', ['#'], None)
        else:
            #Word-medial position, get the segment object on the right
            rhs = word[pos+1]

        e = Environment(lhs, rhs)

        return e

    def config_parse(self, filename):

        if not filename:
            return
        config = configparser.ConfigParser()
        filename = os.path.join(os.getcwd(), 'config', filename)
        if not os.path.exists(filename):
            raise ModelError('Unable to find config file {}'.format(filename))
        config.read_file(open(filename, mode='r', encoding='utf_8_sig'))

        float_words= ['inherent_variation', 'invention_rate', 'samplesize', 'minimum_activation_level', 'min_spread',
                    'minimum_category_distance']
        int_words = ['words_per_turn', 'max_inventions', 'minimum_repetitions', 'initial_lexicon_size',
                    'min_word_length', 'max_word_length', 'generations']

        bool_words = ['allow_unmarked', 'minimize_feature', 'auto_increase_lexicon_size', 'seg_specific_misperceptions']

        for key in config['simulation'].keys():

            value = config['simulation'][key]

            if key == 'seed':
                try:
                    Simulation.seed = int(value)

                except ValueError:
                    Simulation.seed = value

            #not everything in this file should be interpreted as
            #a string
            elif key in int_words:
                value = int(value)
            elif key in float_words:
                value = float(value)
            elif key in bool_words:
                value = config.getboolean('simulation', key)
            elif key == 'initial_inventory':
                value = value.split(',')
                if not len(value) > 1:
                    raise ModelError(
                        ('Initial inventory must be a list of symbols seperated by commas, or else two numbers'
                         ' separated by commas'))

                try:
                    Simulation.initial_inventory = (int(value[0]), int(value[1]))
                except ValueError:
                    Simulation.initial_inventory = ','.join(value)  # needs to be a string for a later function
            elif key == 'max_lexicon_size':
                try:
                    value = config.getboolean('simulation', key)
                    #user supplied False to avoid a limit on the lexicon size
                except ValueError:
                    #user supplied a limit
                    value = int(value)
            elif key == 'initial_words':
                Simulation.initial_words = value.split(',')
            #else:
                #in any other case it's a string, leave it as is


            if hasattr(Simulation,key):
                setattr(Simulation,key,value)

            else:
                raise ModelError('{} is not a Simulation attribute'.format(key))

        misperception_set = list()

        if 'misperceptions' in config.sections():

            for key in config['misperceptions'].keys():
                name = key
                try:
                    filter_,target,change,env,p = config['misperceptions'][key].split(';')
                except ValueError:
                    print('There\'s a semi-colon or comma misplaced in this line: {}:{}'.format(name,config['misperceptions'][key].split(';')))
                    quit()
                misperception_set.append(Misperception(name,filter_,target,change,env,p))

        Simulation.misperception_set = misperception_set


        if 'inventory' in config.sections():
            value = config['inventory']['start']
            value = value.split(',')
            if not len(value) > 1:
                raise ModelError(('Initial inventory must be a list of symbols seperated by commas, or else two numbers'
                ' seperated by commas'))

            try:
                Simulation.initial_inventory = (int(value[0]), int(value[1]))
            except ValueError:
                Simulation.initial_inventory = ','.join(value) #needs to be a string for a later function

        if 'lexicon' in config.sections():
            for key in config['lexicon'].keys():
                if key == 'words':
                    value = config['lexicon']['words']
                    value = value.split(',')
                    Simulation.initial_words = value

                elif key == 'from_sim':
                    value = config['lexicon']['from_sim']
                    file = os.path.join(Simulation.basedir,
                                        'Chapter 5 bigset',
                                        'Simulation output ({})'.format(value),
                                        'temp_output0.txt')
                    with open(file, mode='r', encoding='utf_8_sig') as f:
                        words = list()
                        found_lexicon = False
                        for line in f:
                            if line.startswith('FEATURE'):
                                break
                            if line.startswith('Phoneme'):
                                inventory = line.strip()
                                inventory = inventory.split(':')[-1]
                                Simulation.initial_inventory = inventory
                            elif line.startswith('LEXICON'):
                                found_lexicon = True
                            elif found_lexicon:
                                word = line.strip()
                                if not word:
                                    continue
                                word = word.split(':')[-1]
                                word = word.split('(')[0]
                                word = word.strip()
                                word = word.split('+')
                                words.append(word)
                    Simulation.initial_words = words



    def init_misperceptions(self):

        misperception_set = list()
        try:
            with open(self.misperceptions, encoding='utf-8') as f:
                for line in f:
                    line = line.split(';')
                    name = line[0].strip()
                    filter_ = line[1].split(',')
                    target = line[2].strip()
                    change = line[3].strip()
                    env = line[4].strip()
                    p = line[5].strip()
                    if filter_ == 'ep':
                        pass
                    misperception_set.append(Misperception(name,filter_,target,change,env,p))
        except IOError:
            pass #No misperceptions were specified, so don't make any misperceptions

        Simulation.misperception_set = misperception_set

    def record(self):
        gen = str(self.current_generation)
        filename = 'temp_output{}.txt'.format(gen)
        filename = os.path.join(self.logdir,filename)
        log_file = open(filename, encoding='utf-8', mode='a')

        inventory_message = 'INVENTORY{} ({} segments, including all variants)\r\n'.format(gen, len(self.speaker.inventory))
        print(inventory_message, file=log_file)
        output = list()
        random_seg = random.choice(list(self.speaker.inventory.values()))
        line = '\t{}'.format('\t'.join(list(random_seg.features.keys())))
        output.append(line)
        for seg in self.speaker.inventory:
            features = '\t'.join([v.sign for v in self.speaker.inventory[seg].features.values()])
            line = ''.join([seg,'\t',features])
            output.append(line)
        output.append('\r\n')
        output = '\r\n'.join(output)
        print(output, file=log_file)

        print('\r\nVARIATION\r\n', file=log_file)
        if self.speaker.allophones is not None:
            if isinstance(self.speaker.phonemes[0], Segment):
                phonemes = ','.join([p.symbol for p in self.speaker.phonemes])
            else:
                phonemes = ','.join([p for p in self.speaker.phonemes])
            allophones = list()
            for sr,ur in list(self.speaker.allophones.items()):
                for u in ur:
                    allophones.append('{}~{}'.format(u,sr))
            allophones = ','.join(allophones)

            print('Phonemes:{}\r\nAllophones:{}'.format(phonemes, allophones),
                                                        file=log_file)
        else: #no allophones this turn
            print('Phonemes:{}\r\nAllophones:'.format(self.speaker.inventory), file=log_file)

        print('\r\nLEXICON'+gen+'\r\n', file=log_file)

        for meaning,wordlist in iter(self.speaker.lexicon.items()):
            wordset = list()
            for word in wordlist:
                wordjoin = '+'.join([str(self.speaker.inventory[seg.symbol]) for seg in word])
                wordset.append(''.join([wordjoin, ' (',str(word.freq),')']))
            wordset = ','.join(wordset)
            line = ''.join([str(word.meaning),' :', wordset, '\r\n'])
            print(line, file=log_file)

        print('\r\nFEATURE SPACE'+gen+'\r\n', file=log_file)

        tabs = '\t'*3
        line = tabs.join([str(seg) for seg in self.speaker.inventory.values()])
        line = ''.join(['\t',line,'\r\n'])
        print(line, file=log_file)

        for feature in self.speaker.feature_space:
            printline = [feature]
            append = printline.append#LO
            fs = self.speaker.feature_space[feature]
            for seg in self.speaker.inventory:
                if isinstance(self.speaker, Agent):
                    cloud = [tk for tk in fs if self.speaker.old_to_new_inventory[tk.label] == seg]
                else:
                    cloud = [tk for tk in fs if tk.label == seg]

                min_token = min(cloud)
                min_token = '{value:.4f}'.format(value=min_token.value)
                max_token = max(cloud)
                max_token = '{value:.4f}'.format(value=max_token.value)
                minmax = ' '.join([min_token,'<->',max_token])
                append(minmax)

                #print the distribution
                feature_filename = ''.join(['feature_distributions',gen,'.txt'])
                feature_distribution = open(os.path.join(self.logdir,feature_filename),'a',encoding='utf-8')
                cloud.sort()
                num_bins = 20
                bins = {b*(1/num_bins):0 for b in range(num_bins+1)}
                binsizes = sorted(bins.keys())
                start = 0
                for token in cloud:
                    tokenvalue = token.value
                    for size in range(start, len(binsizes)):
                        if tokenvalue <= binsizes[size]:
                            bins[binsizes[size]] += 1
                            start = size
                            break
                print('{} ({})\r\n'.format(feature, self.speaker.inventory[seg]), file=feature_distribution)
                for b in sorted(bins.keys()):
                    floor = b-(1/num_bins)
                    if floor < 0:
                        floor = 0
                    line = '{}-{}:{}\r\n'.format(floor, b, bins[b])
                    print(line, file=feature_distribution)
                feature_distribution.close()

            append('\r\n')
            printline = '\t'.join(printline)
            print(printline, file=log_file)

        log_file.close()

        if Simulation.seg_specific_misperceptions:
            with open(os.path.join(self.logdir, 'seg_specific_misperceptions.txt'), mode='a', encoding='utf-8') as f:
                print('\nGENERATION {}\n'.format(gen), file=f)
                for misp in Simulation.misperception_set:
                    print(misp.fileprint(), file=f)


    def transmit(self, word):
        """
        This function takes in a Word representing the output of the speaker.
        It applies misperceptions to it (if any apply)
        and returns a Word representing the input for the learner
        """
        soundbite = self.speaker.soundify(word)
        env_list = [self.get_environment(pos, word.string) for pos,seg in enumerate(word.string)]
        change_list = [misp for misp in Simulation.misperception_set if misp.kind == 'change']

        for misperception in change_list:
            for pos,seg in enumerate(word.string):
                if Simulation.seg_specific_misperceptions and not misperception.name.startswith(seg.symbol):
                    applies = False
                else:
                    applies = misperception.apply(env_list[pos],seg)

                if applies:
                    value = soundbite[pos].features[misperception.target].value + misperception.salience
                    if value > 1.0:
                        value = 1.
                    elif value < 0.0:
                        value = 0.
                    soundbite[pos].features[misperception.target].value = value

        return soundbite


    def init_segbase_and_inventory(self):

        path = os.path.join(os.getcwd(), self.features_file)
        Simulation.segbase = Segbase(path, restricted_features='some')

        #check if any set of segments was supplied to the simulation
        #and grab the minimum set of features required to contrast them
        if isinstance(Simulation.initial_inventory, str):
                initial_segs = Simulation.initial_inventory.split(',')
        elif isinstance(Simulation.initial_inventory, tuple):
            C, V = Simulation.initial_inventory
            all_c = [seg.symbol for seg in Simulation.segbase.segments.values() if '-voc' in seg.features.values()]
            all_v = [seg.symbol for seg in Simulation.segbase.segments.values() if '+voc' in seg.features.values()]
            initial_segs = list()
            consonants = random.sample(all_c,C)
            vowels = random.sample(all_v,V)
            initial_segs.extend(consonants)
            initial_segs.extend(vowels)
        else:
            raise ModelError('Wrong value for Simulation.initial_inventory {}'.format(Simulation.initial_inventory))

        Simulation.initial_inventory_list = list()
        for seg in initial_segs:
            Simulation.initial_inventory_list.append(Simulation.segbase.copy_seg(seg))


    def init_max_phonotactics(self):
        """
        Takes a syllable as input and returns all possible sub-syllables
        e.g. the input CVCC should return {V, CV, VC, VCC, CVC, CVCC}
        """

        string = Simulation.phonotactics
        string = string.split(',',1)
        exceptions = False
        if len(string)>1:
            exceptions = string[1:][0]
        string = string[0]
        onsets = ['']
        codas = [''] #for the empty onset and empty coda
        sylist = list()
        for x in string:
            begin,end = string.split('V')
        for n,o in enumerate(begin):
            onsets.append((n+1)*o)
        for n,c in enumerate(end):
            codas.append((n+1)*c)
        for pair in itertools.product(onsets, codas):
            sylist.append('V'.join(pair))
        #sylist.append('V') #all languages must have V as a possible syllable?
        if exceptions:
            exceptions = exceptions.split(',')
            for e in exceptions:
                sylist.remove(e)

        Simulation.phonotactics_list = sylist

    def quit_pyilm(self):
        print('cleaning up...')
        # saving doesn't quite work yet....
        if Simulation.save_sim:
            print('saving...')
            self.speaker.config_file = self.config_file
            self.speaker.current_generation = self.current_generation
            filename = os.path.join(Simulation.basedir, Simulation.save_sim)
            with open(filename, 'wb') as f:
                pickle.dump(self.speaker, f)
        print('...and done! Thank you for using PyILM!')

    def main(self):
        """
        Main loop.
        """
        self.speaker = BaseAgent('0')
        self.listener = Agent('1')

        while self.current_generation < Simulation.generations:
            start = time.time()
            print('on generation {}'.format(self.current_generation))
            # Loop is broken when speaker runs out of things to say
            while 1:
                # produces a Word with Segments with binary Features
                word = self.speaker.talk()

                if word is None:
                    #speaker has run out of words
                    break

                # produces a Word with Sounds with multivalued Tokens
                word = self.transmit(word)

                # inputs a word to listener's learning algorithm
                self.listener.listen(word)

            then = time.time()
            print('Main loop took {}'.format(abs(start-then)))
            print('recording...')
            now = time.time()
            self.record()#records the speaker's information
            then = time.time()
            print('Recording took {}'.format(abs(now-then)))
            now = time.time()
            print('Cleaning up...')
            self.listener.clean_up(self.speaker.inventory)
            self.speaker = self.listener
            self.speaker.choose_words()
            self.generate_seg_specific_misperceptions()
            self.current_generation += 1
            self.speaker.name = 'Speaker '+str(self.current_generation)
            self.listener = Agent('Listener '+str(self.current_generation))
            stop = time.time()
            print('Cleaning up took {}'.format(abs(now-stop)))
            print('This generation took {}'.format(abs(start-stop)))
        self.quit_pyilm()

class BaseAgent(object):
    """
    Agents represent the individuals who learn and use a language.
    A BaseAgent is used in the first generation of the simulation, after that
    only Agents are used. The reason for two classes is that the BaseAgent needs to
    randomly initialize some things, while the Agent needs to learn them.
    """

    def __init__(self, name):
        """
        Initialize 0th generation agent
        """
        self.name = name
        self.algorithm = Simulation.learning_algorithm
        self.phonotactics = Simulation.phonotactics_list
        self.lexicon = Lexicon()
        self.feature_space = FeatureSpace({feature:list() for feature in Simulation.segbase.features})
        self.learned_centroids = list()
        self.inventory = PrintableDictionary()
        self.init_inventory()
        self.init_lexicon()
        self.choose_words()
        self.assign_features()
        self.sample_production_value = self.sample_gaussian
        for meaning in self.lexicon:
            self.init_feature_space(meaning)
        self.phonemes = None
        self.allophones = None


    def choose_words(self, frequency_block=10):
        """
        Deterministically select a list of words to say on this turn
        This replaces a the probabalistic word choice
        see: self.assign_lexicon_probabilities()

        the lexicon is divided up into frequency_block number of blocks, the first
        block has words with frequency 1, and each successive block as twice
        the frequency
        """

        self.production_list = list()
        key_list = reversed(list(self.lexicon.keys()))

#TO MIX UP FREQUENCIES
##        swap_prob = .1
##        for k in range(0, len(key_list)-1, 2):
##            n = random.random()
##            if n <= swap_prob:
##                key_list[k], key_list[k+1] = key_list[k+1], key_list[k]

        for j in range(Simulation.minimum_repetitions):
            n = 0
            for k,meaning in enumerate(key_list):
                self.production_list.extend([meaning for k in range(2**n)])
                if k % frequency_block == 1:
                    n += 1
        random.shuffle(self.production_list)

    def assign_features(self):

        for symbol,seg in iter(self.inventory.items()):
            for feature in seg.features.values():
                sign = seg.features[feature.name].sign
                if sign == '+':
                    maxvalue = 1.
                    if feature.name == 'voc':#prevent overlap of vowels and consonants
                        minvalue = .9
                    else:
                        minvalue = .6

                elif sign == '-':
                    minvalue = 0.
                    if feature.name == 'voc':
                        maxvalue = .1
                    else:
                        maxvalue = .4

                elif sign == 'n':
                    minvalue = 0.
                    maxvalue = 0.

                elif sign == '.':
                    if Simulation.allow_unmarked:
                        minvalue = 0.
                        maxvalue = 1.
                    else:
                        minvalue, maxvalue = random.choice([(0.,0.), (0.,.4), (.6, 1.)])

                mean = random.uniform(minvalue,maxvalue)
                if mean == 0.0 :
                    #this is 'n' value
                    self.inventory[symbol].distributions[feature.name] = (0.0,0.0)
                else:
                    self.inventory[symbol].distributions[feature.name] = (mean,Simulation.min_spread)

    def init_inventory(self):
        for seg in Simulation.initial_inventory_list:
            new_seg = Simulation.segbase.copy_seg(seg.symbol)
            self.inventory[new_seg.symbol] = new_seg

    def new_segment(self,sound,exclude=None):

        if exclude is None:
            exclude = tuple()

        feature_list = list()
        for token in sound.features.values():
            if token.value > .50:
                sign = '+'
            elif token.value <= 0:
                sign = 'n'
            elif token.value <= .5:
                sign = '-'
            elif token.value == '.':
                sign = '.'
            feature_list.append(sign+token.name)

        num = Simulation.segbase.seg_as_int()
        seg = Segment(num, feature_list)
        self.inventory[seg.symbol] = seg
        return self.inventory[seg.symbol]

    def create_distributions(self):

        if Simulation.density_type == 'gaussian':
            self.create_gaussian()
            self.sample_production_value = self.sample_gaussian
        elif Simulation.density_type == 'kde':
            self.create_kde()
            self.sample_production_value = self.sample_kde
        elif Simulation.density_type == 'gaussian_env':
            self.sample_production_value = self.sample_gaussian_env

        else:
            message = 'Simulation density type {} not valid'.format(Simulation.density_type)
            raise ModelError(message)

    def create_gaussian(self):
        samplesize = Simulation.samplesize
        for item in self.inventory:
            segment = self.inventory[item]
            for feature in segment.features:
                fs = self.feature_space[feature]
                dataset = [token.value for token in fs if token.label == segment.symbol]

                if Simulation.medianormean == 'mean':
                    mean = scipy.mean(dataset)
                    std = scipy.std(dataset)
                    if std > 0. and std < Simulation.min_spread:
                        std = Simulation.min_spread
                    self.inventory[item].distributions[feature] = (mean,std)

                elif Simulation.medianormean == 'median':
                    median = scipy.median(dataset)
                    mad = scipy.median([abs(x-median) for x in dataset])
                    if mad < Simulation.min_spread:
                        mad = Simulation.min_spread
                    self.inventory[item].distributions[feature] = (median, mad)


    def sample_gaussian(self,feature,input_word,pos):

        mean, variation = self.inventory[input_word[pos].symbol].distributions[feature]
        value = random.normalvariate(mean, variation)
        if value < 0.:
            value = 0.
        elif value > 1.:
            value = 1.
        return value

    def sample_gaussian_env(self,feature,input_word,pos):

        e = Simulation.get_environment(pos,input_word)
        cloud = [token.value for token in self.feature_space[feature]
                if token.label == input_word[pos].symbol
                and token.env == e]
        if not cloud:
            #this can happen if the invention algorithm generated
            #the input word, and there's a novel environment in it
            #ignore environment this time, but add a token so that later
            #production are influenced by this one
            add_new_token = True
            cloud = [token.value for token in self.feature_space[feature]
                if token.label == input_word[pos].symbol]
        else:
            add_new_token = False

        if Simulation.medianormean == 'mean':
            mean = scipy.mean(cloud)
            std = scipy.std(cloud)
            value = random.normalvariate(mean, std)

        elif Simulation.medianormean == 'median':
            median = scipy.median(cloud)
            mad = scipy.median([abs(token-median) for token in cloud])
            value = random.normalvariate(median, mad)

        if value > 1.:
            value = 1.
        elif value < 0.:
            value = 0.

        if add_new_token:
            self.feature_space[feature].append(Token(feature,value,input_word[pos].symbol,e))

        return value

    def create_kde(self):

        samplesize = Simulation.samplesize
        for item in self.inventory:
            segment = self.inventory[item]
            for feature in self.inventory[item].features:
                cloud = [token.value for token in self.feature_space[feature]
                        if token.label == segment.symbol]
                step = int(round(len(cloud)/(len(cloud)*samplesize)))
                #dataset =  [cloud[pos] for pos,i in enumerate(cloud) if pos%step==0]
                dataset = random.sample(cloud,step)
                try:
                    if len(dataset) == 1:
                        raise LinAlgError
                    self.inventory[item].distributions[feature] = scipy.stats.gaussian_kde(dataset)
                except LinAlgError:
                    #this is raised if every token has the same value.
                    #this happens occasionally with things being all 0.0 values
                    #and rarely all 1.0 values
                    self.inventory[item].distributions[feature] = dataset[0]

    def sample_kde(self,feature,input_word,pos):

        estimator = self.inventory[input_word[pos].symbol].distributions[feature]

        try:
            value = estimator.resample(size=1)
            value = float(value[0][0])
        except AttributeError:
            value = estimator

        if value < 0.:
            value = 0.
        elif value > 1.:
            value = 1.

        return value


    def assign_lexicon_probabilities(self):

        self.lexicon_probs = list()
        append = self.lexicon_probs.append#LO
        N = len(self.lexicon)
        for m,meaning in enumerate(self.lexicon):
            append((N/(m+1), meaning))


    def weighted_sample(self, items, n):

        total = float(sum(w for w, v in items))
        i = 0
        w, v = items[0]
        while n:
            x = total * (1 - random.random() ** (1.0 / n))
            total -= x
            while x > w:
                x -= w
                i += 1
                w, v = items[i]
            w -= x
            yield v
            n -= 1

    def weighted_choice(self, items):

        total = float(sum(w for w, v in items))
        i = 0
        w, v = items[0]
        x = total * (1 - random.random())
        total -= x
        while x > w:
            x -= w
            i += 1
            w, v = items[i]
        w -= x
        return v

    def exemproduce(self, input_word):
        """
        Produce Sound objects out of Segments, using the exemplar space
        """
        sound_list = list()
        append = sound_list.append#LO
        for pos,segment in enumerate(input_word):
            token_list = list()
            segmentfeatures = segment.features#LO
            for feature in segmentfeatures:
                value = self.sample_production_value(feature,input_word,pos)
                #sample_production_value is defined in self.create_distributions
                #it will be one of the functions that starts with sample_
                #e.g. sample_guassian, sample_kde, etc.

                label = input_word[pos].symbol
                token = Token(feature,value,label,None)

                if Simulation.inherent_variation:
                    variation = random.uniform(-1*Simulation.inherent_variation,Simulation.inherent_variation)
                    token.value += variation

                if token.value > 1.:
                    token.value = 1.
                elif token.value < 0.:
                    token.value = 0.
                token_list.append(token)

            sound = Sound(segment.symbol, token_list)
            append(sound)

        return Word(sound_list, meaning=input_word.meaning)

    def soundify(self, input_word):
        """
        Turns a word from the lexicon into a spoken word
        Eventually, a choice of multiple production algorithms
        could appear here
        """
        if Simulation.learning_algorithm == 'exemplars':
            return self.exemproduce(input_word)


    def talk(self,word=None):

        if word is None:
            try:
                meaning = self.production_list.pop(0)
                word_probs = [(word.freq, word) for word in self.lexicon[meaning]]
                word = self.weighted_choice(word_probs)
            except IndexError:
                #all out of words
                word = None

        elif isinstance(word, int):
            wordlist = self.lexicon[word]
            word = random.choice(wordlist)

        return word


    def update_lexicon(self,new_word):
        """
        Add a word to the lexicon if it isn't there already
        """

        new_word.meaning#LO
        try:
            for pos,entry in enumerate(self.lexicon[new_word.meaning]):
                if entry.string == new_word.string:
                    self.lexicon[new_word.meaning][pos].freq += 1
                    break
            else:
                self.lexicon[new_word.meaning].append(new_word)

        except KeyError:
            self.lexicon[new_word.meaning] = [new_word]

    def init_feature_space(self, meaning, initial_deviation=0.005):
        """
        Adds tokens to feature space keeping track of environments
        """

        for i,word in enumerate(self.lexicon[meaning]):
            self.lexicon[meaning][i].freq += 1
            for pos,seg in enumerate(word):
                e = Simulation.get_environment(pos,word)
                for feature in seg.features.values():
                    value = self.sample_production_value(feature.name,word,pos)
                    if value < 0.:
                        value = 0.
                    elif value > 1.:
                        value = 1.
                    token = Token(feature.name, value, seg.symbol, e)
                    self.feature_space[feature.name].append(token)

    def init_lexicon(self):
        """
        First generates a C-V skeleton
        Then randomly places segments into each slot
        """

        unused = list(self.inventory.keys())
        last_word = None
        for word in Simulation.initial_words:
            new_word = list()
            for seg in word:
                new_word.append(self.inventory[seg])
                self.inventory[seg].freq += 1
                try:
                    unused.remove(seg)
                except ValueError:
                    pass
            new_word = Word(new_word)
            for pos, seg in enumerate(new_word.string):
                e = Simulation.get_environment(pos, word)
                self.inventory[word[pos]].update_envs(e)
            self.update_lexicon(new_word)

        if len(self.lexicon) >= Simulation.initial_lexicon_size:
            #this occurs if the user supplied more words than the number given to initial_lexicon_size parameter
            #in this case, we should just add all the words that were supplied to initial_words and ignore the
            #original cap on lexicon size, but we don't generate any more words below. Also, we have to cull the
            #inventory for unused sounds, so this is incompatible with auto_increase_lexicon_size
            if unused:
                inventory = {k: v for k, v in iter(self.inventory.items()) if k not in unused}
                self.inventory = PrintableDictionary(inventory)
            return


        while len(self.lexicon) < Simulation.initial_lexicon_size:
            word = self.coinage(minimal_pair=last_word)
            for seg in word:
                try:
                    unused.remove(seg.symbol)
                except ValueError:
                    pass #we've already removed this value

            if len(self.lexicon)%2 == 0:
                #only invent a minimal pair every other time,
                #otherwise the whole inventory looks like the very first word
                last_word = word
            else:
                last_word = None


        for unused_seg in unused[:]:
            used = sorted([s for s in self.inventory.values() if s.freq > 1], key=lambda x: x.freq, reverse=True)
            if not used:
                #no segments have enough frequency to be in this list, so jump the loop.
                #other solutions exist below
                break
            made_replacement = False
            counter = 0 #to avoid infinite loops
            while True:
                key = random.choice(list(self.lexicon.keys()))
                random_word = self.lexicon[key][0]
                if self.inventory[unused_seg].is_vowel():
                    segs = [(pos,seg) for (pos,seg) in enumerate(random_word) if random_word[pos].is_vowel()]
                else:
                    segs = [(pos,seg) for (pos, seg) in enumerate(random_word) if not random_word[pos].is_vowel()]
                if not segs:
                    continue
                for pos,seg in segs:
                    if seg in used:
                        random_word[pos] = self.inventory[unused_seg]
                        unused.remove(unused_seg)
                        self.inventory[unused_seg].freq += 1
                        self.inventory[seg.symbol].freq -= 1
                        made_replacement = True
                        break
                counter += 1
                if counter >= 50:
                    #we've tried too many times and nothing happened. break this loop
                    #and deal with the leftover segments in the next loop below
                    made_replacement = True
                if made_replacement:
                    break


        if unused and Simulation.auto_increase_lexicon_size:
            #in this case, it's more important to use all of the initial segments at least once
            last_word = None
            for seg in unused[:]:
                word = self.coinage(minimal_pair=last_word, include=seg)
                last_word = word
                unused.remove(seg)
        else:
            #in this case, it's more important to keep the lexicon a particular size
            #even if you don't use all the segments
            #not using all of the initial segments can cause weird behaviour with the
            #coinage function, because the inventory still contains references to the
            #unsued sounds. this step filters the inventory to prevent any problems later
            inventory = {k:v for k,v in iter(self.inventory.items()) if k not in unused}
            self.inventory = PrintableDictionary(inventory)

        return

    def calculate_seg_freq(self, seg):

        total_seg_count = 0
        count = 0
        symbol = seg.symbol#LO
        for meaning in self.lexicon:
            for word in self.lexicon[meaning]:
                total_seg_count += len(word)
                for cmp_seg in word:
                    if cmp_seg.symbol == symbol:
                        count += 1
        freq = count/total_seg_count
        freq = freq * 10
        freq = int(round(freq))
        if freq < 1:
            freq = 1
        return freq

    def coinage(self, minimal_pair=None, include=None):
        """
        Invent a new word
        If minimal_pair is None, the invented word is completely random. If a Word is supplied as a value
        then the invented word is a minimal pair with it.
        If include is None, there are no restriction on which sounds go into the word. If a Segment is supplied
        as a value, the invented word will include this segment somehwere. Only one seg is allowed.
        As a side-effect, this function adds the word to the lexicon as well
        """
        choice = random.choice#LO
        get_environment = Simulation.get_environment#LO
        word = list()
        cons = [s for s in self.inventory if self.inventory[s].features['voc'] == '-voc']
        vowels = [s for s in self.inventory if self.inventory[s].features['voc'] == '+voc']

        if minimal_pair is None:
            syl_length = random.randint(Simulation.min_word_length,
                                    Simulation.max_word_length)
            for j in range(syl_length):
                #Seed the words with random segments according to some
                #phonotactics
                syl_type = choice(self.phonotactics)
                for x in syl_type:
                    if x == 'C':
                        seg = choice(cons)
                    elif x == 'V':
                        seg = choice(vowels)
                    word.append(self.inventory[seg])
                    self.inventory[seg].freq += 1

        else:
            rand_pos = random.randint(0,len(minimal_pair))
            for pos,seg in enumerate(minimal_pair):
                seg = seg.symbol
                if pos == rand_pos:
                    try:
                        if seg in cons:
                            random_seg = random.choice([c for c in cons if not c==seg])
                        else:
                            random_seg = random.choice([v for v in vowels if not v==seg])
                        word.append(self.inventory[random_seg])
                        self.inventory[random_seg].freq += 1
                    except IndexError:
                        #can't choose from an empty sequence
                        #this happens when there's just single vowel
                        #or a single consonant in the inventory
                        #only occurs in small toy simulations
                        word.append(self.inventory[seg])
                        self.inventory[seg].freq += 1
                else:
                    word.append(self.inventory[seg])
                    self.inventory[seg].freq += 1

        if include is not None:
            if isinstance(include, Segment):
                include = include.symbol
            need_vowel = self.inventory[include].is_vowel()
            pos_list = [n for n in range(len(word))]
            random.shuffle(pos_list)
            while pos_list:
                pos = pos_list.pop()
                if word[pos].is_vowel() and need_vowel:
                    word[pos] = self.inventory[include]
                    self.inventory[include].freq += 1
                    break
                else:
                    word[pos] = self.inventory[include]
                    self.inventory[include].freq += 1
                    break

        word = Word(word)
        for pos,seg in enumerate(word.string):
            e = get_environment(pos,word)
            self.inventory[word[pos].symbol].update_envs(e)
        self.update_lexicon(word)

        return word


class Agent(BaseAgent):
    """
    Subclass of BaseAgent for use in generation 1 onwards. Use BaseAgent for
    generation 0.
    """

    def __init__(self,name):

        self.name = name
        self.feature_space_counter = 0
        self.phonotactics = Simulation.phonotactics_list
        self.lexicon = Lexicon()
        self.feature_space = FeatureSpace()
        for feature in Simulation.segbase.features:
            self.feature_space[feature] = list()
        self.learned_centroids = list()
        self.inventory = PrintableDictionary()
        self.fuzzy_learning = False
        self.min_activation = math.e**-(1-Simulation.minimum_activation_level)
        self.current_turn = 0
        self.choose_words()
        self.phonemes = None
        self.allophones = None

    def make_binary_features(self, quorum=.5,cutoff=.5):

        if Simulation.feature_division == 'kmeans':
            self.kmeans_features(quorum)
        elif Simulation.feature_division.startswith('abs'):
            self.absolute_features(quorum,cutoff)
        else:
            raise ModelError('Unknown feature-partitioning algorithm: {}'.format(Simulation.feature_division))

        return

    def absolute_features(self, quorum, cutoff = .5):
        """
        Coarsely groups features
        """
        for feature in self.feature_space:
            tokens = self.feature_space[feature]
            minus_cluster = [tk for tk in tokens if tk.value<=cutoff]
            plus_cluster = [tk for tk in tokens if tk.value>cutoff]
            for seg in self.inventory:
                minus_count = len([1 for tk in minus_cluster if tk.label == seg])
                plus_count = len([1 for tk in plus_cluster if tk.label == seg])
                if minus_count/(minus_count+plus_count) >= quorum:
                    sign = '-'
                else:
                    sign = '+'
                if Simulation.allow_unmarked and all([tk.value<=0 for tk in minus_cluster]):
                    sign == 'n'

                self.inventory[seg].features[feature].sign = sign


    def kmeans_features(self,k=2,partition_inventory=True):

        sorted_features = sorted(list(self.feature_space.keys()))

        if partition_inventory:
            cons_data = array([[tk.value for tk in self.feature_space[feature]
                                if self.inventory[tk.label].features['voc']=='-voc']
                        for feature in sorted_features])
            vowel_data = array([[tk.value for tk in self.feature_space[feature]
                                 if self.inventory[tk.label].features['voc']=='+voc']
                        for feature in sorted_features])
            cons_clusters = Cluster.kmeans(cons_data, k)
            a,b = cons_clusters[0]
            if sum(a)>sum(b):
                plus_cons = a
                minus_cons = b
            else:
                plus_cons = b
                minus_cons = a
            vowel_clusters = Cluster.kmeans(vowel_data, k)
            a,b = vowel_clusters[0]
            if sum(a)>sum(b):
                plus_vowel = a
                minus_vowel = b
            else:
                plus_vowel = b
                minus_vowel = a

        else:
            all_clusters = Cluster.kmeans(array([[tk.value
                            for tk in self.feature_space[feature]]
                                for feature in sorted_features]),2)
            a,b = all_clusters[0]
            if sum(a)>sum(b):
                plus_all = a
                minus_all = b
            else:
                plus_all = b
                minus_all = a

        for seg in self.inventory:

            if partition_inventory:
                if self.inventory[seg].features['voc'] == '-voc':
                    minus_cluster = minus_cons
                    plus_cluster = plus_cons
                else:
                    minus_cluster = minus_vowel
                    plus_cluster = plus_vowel
            else:
                minus_cluster = minus_all
                plus_cluster = plus_all


            for j,feature in enumerate(sorted_features):
                minus_centroid = minus_cluster[j]
                plus_centroid = plus_cluster[j]
                data = [tk.value for tk in self.feature_space[feature] if tk.label==seg]

                if abs(minus_centroid-plus_centroid) < Simulation.minimum_category_distance:
                    #kmeans centroids are "too close"
                    if max([minus_centroid, plus_centroid]) < .5:
                        sign = '-'
                    else:
                        sign = '+'
                else:
                    minus_distance = sum([(value-minus_centroid)**2 for value in data])
                    plus_distance = sum([(value-plus_centroid)**2 for value in data])
                    if minus_distance <= plus_distance:
                        sign = '-'
                    else:
                        sign = '+'

                self.inventory[seg].features[feature].sign = sign


    def exemplearn(self,sound,pos,meaning):
        """
        Returns a Segment object

        Creates a matrix like this:
            f1  f2  f3  f4  ...
        /p/ 12  15  0   9
        /b/ 12  34  12  98
        /k/ 55  24  8   9
        /g/ 67  30  6   95
        ...                 ...

        where each entry i,j represents the summed activation of all tokens from
        that segment along that dimentions, with respect to the input token
        Make the input Sound a member of the category (row) with the highest sum
        """

        #activation_matrix = collections.defaultdict(list)
        activation_matrix = dict()

        for seg in self.inventory:
            activation_matrix[seg] = list()
            append = activation_matrix[seg].append#LO
            check_space = self.feature_space['voc']#LO
            k = len([1 for tk in check_space if tk.label == seg])
            #k is constant for all dimensions, checking any dimension would work
            for token in sound.features.values():
                total_activation = sum(E**-abs(token.value-other.value)
                                        for other in self.feature_space[token.name]
                                    if other.label == seg)/k
                if total_activation < self.min_activation:
                    #fails to meet minimum activation on some dimension
                    #so remove entirely from consideration
                    garbage = activation_matrix.pop(seg)
                    break
                else:
                    append(total_activation)

            else:
                #part of for-else
                #reached if the segment meets min_activation on all dimensions
                append(sum(activation_matrix[seg])/len(activation_matrix[seg]))

        if activation_matrix:
            best_matches = [(v[-1],k) for k,v in iter(activation_matrix.items())]
            best_matches.sort()
            best_match = best_matches[-1]
            seg = self.inventory[best_match[1]]
            self.inventory[best_match[1]].freq += 1

        else:
            #activation_matrix is empty
            #could be the first turn, so there's no inventory yet
            #could be that nothing was phonetically similar enough
            seg = self.new_segment(sound, self.inventory)
            self.inventory[seg.symbol].freq += 1

        #add some temporary tokens to feature space so that anything so far in
        #the word can influence learning in the rest of the word.
        #they have an env=None because we don't know the full environmnets yet
        #and env doesn't influence future learning
        for feature in self.feature_space:
            token = Token(feature,sound.features[feature].value,seg.symbol,None)
            self.feature_space[feature].append(token)

        return seg

    def remove_duplicate_segments(self, verbose=False):

        duplicates = list()
        for seg1, seg2 in itertools.combinations(self.inventory, 2):
            if self.inventory[seg1].features == self.inventory[seg2].features:
                duplicates.append((seg1, seg2))

        for pair in duplicates:

            if pair[0] not in self.inventory:
                if pair[1] not in self.inventory:
                    continue#this occurs in rare cases of triplicates
                winner = pair[1]
                loser = pair[0]
            elif pair[1] not in self.inventory:
                if pair[0] not in self.inventory:
                    continue#this occurs in rare cases of triplicates
                winner = pair[0]
                loser = pair[1]
            elif self.inventory[pair[0]].freq > self.inventory[pair[1]].freq:
                winner = pair[0]
                loser = pair[1]
            else:
                loser = pair[0]
                winner = pair[1]
            if verbose:
                print('winner={}, loser={}'.format(winner, loser))

            #update feature_space
            for feature in self.feature_space:
                for pos,token in enumerate(self.feature_space[feature]):
                    if token.label == loser:
                        self.feature_space[feature][pos].label = self.inventory[winner].symbol
                    if token.env.rhs.symbol == loser:
                        self.feature_space[feature][pos].env.rhs = self.inventory[winner]
                    if token.env.lhs.symbol == loser:
                        self.feature_space[feature][pos].env.lhs = self.inventory[winner]


            #update lexicon
            for word in self.lexicon.get_full_lexicon():
                for pos,seg in enumerate(word.string):
                    if seg.symbol == loser:
                        word.update_string(pos,self.inventory[winner])

            #update inventory
            try:
                self.inventory[winner].freq += self.inventory[loser].freq
                del self.inventory[loser]
            except KeyError as e:
                #this occurs in rare cases of triplicates
                if verbose:
                    print('Could not find {}. Loser was {}. Winner was {}'.format(e, loser, winner))

            #update segment envs
            for seg in self.inventory:
                for env in self.inventory[seg].envs:
                    e = self.inventory[seg].envs[env]
                    if e.rhs.symbol == loser:
                        self.inventory[seg].envs[env].rhs = self.inventory[winner]
                    if e.lhs.symbol == loser:
                        self.inventory[seg].envs[env].lhs = self.inventory[winner]

        return

    def pick_epenthetic_vowel(self):
        """NOT IMPLEMENTED"""
        vowels = [seg for seg in self.inventory if seg.features['voc'] == '+']
        most_freq = sorted(vowels, key=lambda x:x.freq)[-1]
        self.epenthesis = self.inventory[most_freq]

    def invention(self):
        """
        Potentially create a set of new words to be introduced this generation.
        Probability of invention is set by Simulation.invention_rate
        """
        for j in range(Simulation.max_inventions):
            n = random.random()
            if n <= Simulation.invention_rate:
                m = random.random()
                if m <= Simulation.make_minimal_pair:
                    k = random.choice(list(self.lexicon.keys()))
                    min_pair = random.choice(self.lexicon[k])
                    new_word = self.coinage(minimal_pair=min_pair)
                else:
                    new_word = self.coinage(minimal_pair=None)

                if Simulation.max_lexicon_size and len(self.lexicon)+1 > Simulation.max_lexicon_size:
                    self.replace_lexicon(new_word)
                else:
                    self.update_lexicon(new_word)

    def replace_lexicon(self, new_word):
        """
        Replace a word in the lexicon with a new one that has been invented.
        This is only called if the Simulation has been run with a max_lexicon_size
        Low frequency words are targeted for removal.
        """
        n = 0
        freq = 1
        while 1:
            n += 1
            meaning = random.choice(list(self.lexicon.keys()))
            if sum(word.freq for word in self.lexicon[meaning]) < freq:
                self.lexicon.pop(meaning)
                break
            if n >= len(self.lexicon):
                freq += 1

        self.update_lexicon(new_word)

    def clean_up(self, last_inventory):
        """
        Called at the end of each Simulation.main() loop to finish up some
        learning algorithm issues and to create the phonetic distributions
        for the next turn as speaker
        """
        self.make_binary_features()
        self.remove_duplicate_segments()
        self.assign_lexicon_probabilities()
        self.create_distributions()
        self.invention()
        self.relabel_inventory(last_inventory)
        self.look_for_allophones()

        self.current_turn += 1

    def look_for_allophones(self):

        variants = collections.defaultdict(list)
        for meaning in self.lexicon:
            if not len(self.lexicon[meaning]) > 1:
                continue

            word_list = list(set(self.lexicon[meaning]))
            for n in range(len(word_list[0])):#all Words are the same length
                comp_segs = [word[n] for word in word_list]
                if not all(cs==word_list[0][n] for cs in comp_segs):
                    for seg in comp_segs:
                        variants[seg.symbol].extend([
                            (s,meaning,Simulation.get_environment(n,word_list[i]))
                            for (i,s) in enumerate(comp_segs)
                            if not s == seg])

        self.phonemes = [seg.symbol for seg in self.inventory.values() if not seg.symbol in variants]
        self.allophones = dict()

        for variant in variants:
            alts = list(set([x[0].symbol for x in variants[variant]]))
            words = [x[1] for x in variants[variant]]
            variant = self.inventory[variant]
            if any(variant in word for word in self.lexicon.get_full_lexicon(exclude=words)):
                #if there are any words where the variant doesn't alternate
                #then call it a phoneme
                self.phonemes.append(variant.symbol)
            else:
                #if it only ever appears as a variant of something, call it
                #an allophone
                self.allophones[variant.symbol] = alts

        for allophone in self.allophones:#gets a string
            for a in self.allophones[allophone]:#gets a Segment
                if a in self.phonemes:
                    self.allophones[allophone] = [a]
                    break
                    #there's at least one variant which is not itself just an allophone
                    #make that one the underlying form (this may not be phonologically
                    #very accurate, but it suffices for counting inventory size)
            else:
                #if we don't hit break, all variants are themselves allophones
                #pick one of these sounds to go into the inventory
                self.phonemes.append(allophone)
                self.allophones[allophone] = list()

        self.phonemes = [self.inventory[p] for p in self.phonemes]
        return

    def relabel_inventory(self, last_inventory):
        self.old_to_new_inventory = dict()
        exclude = list()
        for old_seg in last_inventory.values():
            for new_seg in self.inventory.values():
                if new_seg.symbol in exclude:
                    continue
                if new_seg == old_seg:
                    self.old_to_new_inventory[new_seg.symbol] = old_seg.symbol
                    exclude.append(new_seg.symbol)

        remainder = [seg for seg in self.inventory.values() if not seg.symbol in exclude]
        # consult segbase for any remaining segments
        for seg in remainder:
            new_seg = Simulation.segbase.choose_symbol_from_features(list(seg.features.values()),
                                           exclude=list(self.old_to_new_inventory.values()))
            self.old_to_new_inventory[seg.symbol] = new_seg.symbol
            exclude.append(new_seg.symbol)

        for seg in list(self.inventory.keys()):
            seg_info = self.inventory.pop(seg)
            new_symbol = self.old_to_new_inventory[seg]
            self.inventory[new_symbol] = seg_info
            self.inventory[new_symbol].symbol = new_symbol


        return

    def assign_lexicon_probabilities(self):
        self.lexicon_probs = list()
        append = self.lexicon_probs.append#LO
        N = len(self.lexicon)
        for m,meaning in enumerate(self.lexicon):
            append((N/(m+1), meaning))

    def listen(self, word):
        """
        The listening phase consists of three other events:
            the listener parses the input into Segments
            the listener updates the lexicon
            the listener updates the inventory
        """

        get_environment = Simulation.get_environment

        if Simulation.learning_algorithm == 'exemplars':

            segmented_word = list()

            for pos,sound in enumerate(word.string[:]):
                segment = self.exemplearn(sound,pos,word.meaning)#this will also update the inventory
                segmented_word.append(segment)
            segmented_word = Word(segmented_word, word.meaning)

            #now update feature space
            for pos,seg in enumerate(segmented_word.string):
                e = get_environment(pos,segmented_word)
                wordfeatures = word[pos].features
                for feature in wordfeatures:
                    token = wordfeatures[feature]
                    self.feature_space[feature].append(Token(token.name,token.value,seg.symbol,e,time=self.current_turn))

            #There are some "temporary" tokens in feature space that need to be cleared out
            for feature in self.feature_space:
                self.feature_space[feature] = [tk for tk in self.feature_space[feature] if tk.env is not None]

            self.update_lexicon(segmented_word)

        else:
            #Other learning algorithms could get added here in the future
            raise ModelError('{} is not a valid learning algorithm'.format(Simulation.learning_algorithm))

if __name__ == '__main__':
    try:
        config_file = sys.argv[1]

        if config_file == 'False' or config_file == 'None':
            config_file = False #use defaults

    except IndexError:
        config_file = False #use defaults

    if config_file:
        if config_file.endswith('.ini'):
            s = Simulation(config_file=config_file)
        else:
            print(config_file)
            raise IOError('File is not a recognized type. Must be .ini Config file')
    else:
        s = Simulation(config_file=config_file)

    s.main()