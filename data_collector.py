import os
import collections
import phonology

class DataCollector(object):
    """
    Object for gathering information about simulation results.
    """

    @classmethod
    def get_economy_scores(cls, folder, sim):
        """
        Returns a dictionary where keys are economy metrics, and values are a list of scores
        for each generation in a simulation.
        """
        path = os.path.join(folder, 'simulation output ({})'.format(sim), 'simulated economy scores ur.txt')
        if not os.path.exists(path):
            return None
        economy_scores = {'simple': list(), 'frug': list(), 'exp': list(), 'rel':list()}
        with open(path, mode='r') as f:
            f.readline()
            for line in f:
                scores = [float(x.strip()) for x in line.split(',')[:4]]
                economy_scores['simple'].append(scores[0])
                economy_scores['frug'].append(scores[1])
                economy_scores['exp'].append(scores[2])
                economy_scores['rel'].append(scores[3])

        return economy_scores

    @classmethod
    def get_inventory(cls, gen, folder, inv_type='both', return_type='segment', cons_only=True):
        """
        Returns a dictionary where keys are segment symbols, and values are
        a list of phonological features.
        inv_type can be 'underlying', 'surface', or 'both' (which returns the sum
        of the two, it doesn't return two values)
        Return None if no appropriate temp_output file can be found in the folder
        """

        inventory = list()
        filename = 'temp_output{}.txt'.format(gen)
        if not os.path.exists(os.path.join(folder,filename)):
            return None
        with open(os.path.join(folder, filename), encoding='utf_8_sig') as f:
            f.readline()
            f.readline()#dunno why i have to do this twice...
            feature_names = f.readline()
            feature_names = feature_names.strip()
            feature_names = feature_names.split('\t')

            for line in f:
                line = line.strip()
                if (not line) or line.startswith('VAR'):
                    continue
                elif line.startswith('Phonemes'):
                    phonemes = line.split(':')[-1]
                    phonemes = phonemes.split(',')
                elif line.startswith('Allophones'):
                    allophones = line.split(':')[-1]
                    allophones = allophones.split(',')
                    allophones = [a.split('~')[-1] for a in allophones]
                    phonemes = [p for p in phonemes if not p in allophones]
                    break
                else:
                    inventory.append(line) #this creates a list of segments with phonological features values


        if return_type == 'segment':
            new_inventory = dict()
            for line in inventory:
                line = line.split('\t')
                symbol = line[0]
                features = [sign+name for sign,name in zip(line[1:],feature_names)]
                if inv_type in ['underlying', 'core', 'ur', 'UR'] and symbol in phonemes:
                    new_inventory[symbol] = features
                elif inv_type in ['surface', 'sr', 'SR', 'phonetic'] and symbol not in phonemes:
                    new_inventory[symbol] = features
                elif inv_type == 'both':
                    new_inventory[symbol] = features

        elif return_type == 'pyilm':
            new_inventory = list()
            for line in inventory:
                line = line.split('\t')
                symbol = line[0]
                features = [sign+name for sign,name in zip(line[1:], feature_names)]
                new_inventory.append(phonology.Segment(symbol, features))

        elif return_type == 'string':
            new_inventory = [line.split('\t')[0] for line in inventory]
            if inv_type in ['underlying', 'core', 'ur', 'UR']:
                new_inventory = [seg for seg in new_inventory if seg in phonemes]
            elif inv_type in ['surface', 'sr', 'SR', 'phonetic']:
                new_inventory = [seg for seg in new_inventory if not seg in phonemes]
            #else inv_type=='both', just return the new_inventory variable

        return new_inventory

    @classmethod
    def get_phonotactics(cls, folder):

        shape = None
        for file in os.listdir(folder):
            if file.endswith('.ini'):
                config = os.path.join(folder, file)
                break
        else:
            raise FileNotFoundError('Cannot find .ini file anywhere in {}'.format(folder))

        with open(config, encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('phonotactics'):
                    shape = line.split('=')[-1].strip()
                    break

        if shape is None:
            raise AttributeError('This simulation has no phonotactic value in its config file')

        return shape

    @classmethod
    def get_variants(cls, gen, folder):
        """
        Returns (phonemes,allophones)
        phonemes    list of phonemes in inventory as strings
        allophones  dict of {surface:underlying}
        """
        filename = 'temp_output{}.txt'.format(gen)

        with open(os.path.join(folder, filename), encoding='utf_8_sig', mode='r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('Phonemes'):
                line = line.strip()
                phonemes = line.split(':')[-1].split(',')
            if line.startswith('Allophones'):
                allophones = dict()
                line = line.strip()
                line = line.split(':')[-1]
                if not line:
                    pass #no variation this turn
                else:
                    line = line.split(',')
                    for pair in line:
                        ur,sr = pair.split('~')
                        allophones[sr] = ur

        return phonemes,allophones


    @classmethod
    def get_feature_distributions(cls,gen,folder,chosen_seg):
        """
        Reads a feature_distribution file output by PyILMs for the current
        generation and simulation number, and returns a dictionary of the
        binned results

        Args
        ----
        gen         int or str, current generation
        folder      str, path to current output folder
        chosen_seg  str, current segment selected

        Returns
        -----
        features    dict of dicts, {feature: {bin:x}} where x is then number of exemplars that
                        fall inside that bin range

        """

        filename = 'feature_distributions{}.txt'.format(str(gen))
        path = os.path.join(folder,filename)
        try:
            with open(path, mode='r', encoding='utf-8 sig') as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return 'error'

        features = collections.defaultdict(dict)
        foundit = False
        for line in lines:
            if line == '' or line == '\n':
                continue
            line = line.split('(')
            if len(line)>1:

                if line[1].rstrip(')') == chosen_seg:
                    foundit = True
                    feature = line[0].strip()
                    feature = feature.strip('\ufeff')
                else:
                    foundit = False
            else:
                line = line[0]
                if line[0].isdigit() and foundit:
                    bin_,value = line.split(':')
                    bin_ = float(bin_.split('-')[0])
                    value = int(value)
                    features[feature][bin_] = value
        return features

    @classmethod
    def get_inventory_sizes(cls, folder, everything=True):
        """
        Returns list of integers where the nth number is the size of the inventory
        at generation n of the simulation in folder
        if everything==True, return a count of surface+allopones
        else return just underlying sounds
        """
        inv_sizes = list()
        n = 0
        while 1:
            filename = 'temp_output{}.txt'.format(n)
            path = os.path.join(folder, filename)
            try:
                with open(path,encoding='utf-8',mode='r') as f:
                    for line in f:
                        if everything and line.startswith('INV'):
                            inv_sizes.append(int(line.split('(')[1].split(')')[0].split(' ')[0]))
                            break
                        elif not everything and line.startswith('Phonemes'):
                            inv_sizes.append(len(line.split(':')[-1].split(',')))
                            break
                n +=1
            except IOError:
                break

        return inv_sizes

    @classmethod
    def get_lexicon(cls, gen, folder):
        """
        Returns a dictionary lexicon[meaning]= [word1,word2,...]
        """

        try:
            with open(os.path.join(folder,'temp_output{}.txt'.format(gen)), 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return 'error'

        lexicon = dict()
        found_lexicon = False
        for line in lines:
            line = line.strip()
            if not line:
            #if line == '\n' or line == '':
                continue
            if line.startswith('FEA'):
                break
            if found_lexicon:
                line = line.split(':')
                lexicon[int(line[0].strip())] = line[1]
            elif line.startswith('LEX'):
                found_lexicon = True
        return lexicon

    @classmethod
    def get_phonetic_feature_limits(cls, gen, folder, seg):

        with open(os.path.join(folder,'temp_output{}.txt'.format(gen)), 'r', encoding='utf_8_sig') as f:
            lines = [line.strip() for line in f.readlines()]
        found_features = False
        features = dict()
        pos = 0
        collect_features = list()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('OLD '):
                #gone past the end of the features part
                segs = collect_features[0]
                segs = segs.split()
                segs = [s.strip() for s in segs]
                for s in range(len(segs)):
                    if segs[s] == seg:
                        pos = s
                        break

                for line in collect_features[1:]:#first line is a list of segments
                    line = line.split('\t')
                    feature = line[0]
                    values = line[1:]
                    value = values[pos] #get the feature value for the correct segment
                    features[feature] = value
                break

            if found_features:
                line = line.strip()
                if line:
                    collect_features.append(line)

            elif line.startswith('FEAT'):
                found_features = True

        return features