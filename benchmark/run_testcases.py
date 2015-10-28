#!/usr/bin/env python

import os
import sys

if __name__ == '__main__':
    if len(sys.argv[1:]) <= 0:
        print 'usage: %s <platform_name>' % sys.argv[0]
        exit(1)

    platform_name = sys.argv[1]
    print 'running on %s' % platform_name
    
    print 'running test case 1: different thread number...'
    thr_steps = 10
    thr_start = 10
    thr_end   = 240

    i = thr_start
    while i <= thr_end:
        cmd_str = ('./build/msbeam -f ./data/phant_g1.dat -t %d -m %s' 
                % (i, platform_name))
        print cmd_str
        os.popen(cmd_str)
        i += 10
