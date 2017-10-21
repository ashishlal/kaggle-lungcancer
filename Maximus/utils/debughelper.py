import sys

# Be safe and define a maximum of frames we're trying to walk up
MAX_FRAMES = 20

def save_to_interactive(dct):
    n = 0
    # Walk up the stack looking for '__name__'
    # with a value of '__main__' in frame globals
    for n in range(MAX_FRAMES):
        cur_frame = sys._getframe(n)
        name = cur_frame.f_globals.get('__name__')
        if name == '__main__':
            # Yay - we're in the stack frame of the interactive interpreter!
            # So we update its frame globals with the dict containing our data
            cur_frame.f_globals.update(dct)
            break
