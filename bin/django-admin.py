#!/usr/bin/env python3.6

import os; activate_this=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'activate_this.py'); exec(compile(open(activate_this).read(), activate_this, 'exec'), { '__file__': activate_this}); del os, activate_this

from django.core import management

if __name__ == "__main__":
    management.execute_from_command_line()