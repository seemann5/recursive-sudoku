
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def cR(arg):
    return Colors.FAIL + str(arg) + Colors.ENDC
def cG(arg):
    return Colors.OKGREEN + str(arg) + Colors.ENDC
def cB(arg):
    return Colors.OKBLUE + str(arg) + Colors.ENDC
def cY(arg):
    return Colors.WARNING + str(arg) + Colors.ENDC

def color_outcome(outcome: int):
    if outcome == 1:
        return cR(1)
    elif outcome == 2:
        return cG(2)
    elif outcome == 3:
        return cB(3)
    elif outcome == 4:
        return cY(4)
    else:
        raise ValueError