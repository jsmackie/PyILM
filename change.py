#!/usr/bin/env python

import random
import phonology

class Misperception(object):
    """
    name is a string, for example 'word final devoicing'

    filter is a list of feature names, e.g. '+voice' describing segments affected by this misperception

    target is the name of the feature that changes if the misperception applies

    salience is the amount by which it changes, should be a float in the interval [0,1]

    env is a string describing where the change applies, with contexts described at a feature level,
    e.g. '+cont_#', use the symbol '*' for context-free changes

    p is the probability with which the misperception applies. Should be a float in the interval [0,1]
    """

    def __init__(self, name, filter_, target, salience, env, p):
        self.name = name
        self.filter = filter_ if isinstance(filter_, list) else filter_.split(',')
        self.target = target
        self.p = float(p)

        try:
            self.salience = float(salience)
        except ValueError:
            #it's a string
            self.salience = salience
        if isinstance(env, phonology.Environment):
            self.env = env
        elif env == '*':
            #context-free
            lhs = phonology.EmptySegment('NULL',[])
            rhs = phonology.EmptySegment('NULL',[])
            self.env = phonology.Environment(lhs, rhs)
        else:
            #context-sensitive
            lhs,rhs = env.split('_')
            lhs_name = lhs
            rhs_name = rhs
            lhs = lhs.split(',')
            rhs = rhs.split(',')
            lhs = phonology.Segment(lhs_name, lhs) if lhs[0] else phonology.EmptySegment('NULL',[])
            rhs = phonology.Segment(rhs_name, rhs) if rhs[0] else phonology.EmptySegment('NULL',[])
            self.env = phonology.Environment(lhs,rhs)


        if self.filter[0] == 'del':
            self.kind = 'del'
            self.target = self.target.split(',')
        elif self.filter[0] == 'ep':
            self.kind = 'ep'
            self.target = phonology.Segment(self.target, self.target.split(','))
        else:
            self.kind = 'change'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{} ({}, {})'.format(self.name, self.salience, self.target)

    def fileprint(self):
        #useful for printing a misperception in the config file format
        return '{}={}'.format(self.name,
                              ';'.join([str(x) for x in [self.filter,self.target,self.salience,self.env,self.p]]))

    def apply(self,env,seg):

        n = random.random()
        if n > self.p:
            return False #no luck this time
        if not (self.env.lhs < env.lhs and self.env.rhs < env.rhs):
            return False#does not apply in this environment

        if self.kind == 'del':
            if set(self.target).issubset(set([str(f) for f in seg.features.values()])):
                return True #means 'should delete this seg'
            else:
                return False

        elif self.kind == 'ep':
            #since the surrounding environment matched (checked in the above
            #if-statement), we're good to go. no need to check segments because
            #we're inserting one
            return True

        elif self.kind == 'change':
            if all(x in seg.feature_list for x in self.filter):
                return True

        return False