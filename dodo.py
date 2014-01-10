
COVER_OPTIONS = '--with-coverage --cover-package bem --cover-package rotor --cover-html --cover-html-dir=../cover'
NOSE_OPTIONS = '%s' % COVER_OPTIONS

def task_test():
    return {
        'actions': ['spec %s' % NOSE_OPTIONS]
    }
