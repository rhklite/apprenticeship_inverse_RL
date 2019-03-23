import gc
import torch
import inspect
#
# Color terminal (https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python).
class Colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#
# Error information.
def lineInfo():
    callerframerecord = inspect.stack()[2]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s::%s:%d' % (file, info.function, info.lineno)
#
# Line information.
def getLineInfo(leveloffset=0):
    level = 2 + leveloffset
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s: %d' % (file, info.lineno)
#
# Colours a string.
def colourString(msg, ctype):
    return ctype + msg + Colours.ENDC
#
# Print something in color.
def printColour(msg, ctype):
    print(colourString(msg, ctype))
#
# Print information.
def printInfo(*umsg):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in umsg:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.OKGREEN) + lst
    print(msg)
#
# Print error information.
def printFrame():
    print(lineInfo(), Colours.WARNING)
#
# Print an error.
def printError(*errstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in errstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.FAIL) + lst
    print(msg)
#
# Print a warning.
def printWarn(*warnstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in warnstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.WARNING) + lst
    print(msg)
#
# Print information about a tensor.
def printTensor(tensor, usrmsg='', leveloffset=0):
    if isinstance(tensor, torch.Tensor):
        msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': (' + colourString(str(tensor.dtype) + ' ' + str(tensor.device), Colours.WARNING) + ') -- '  + colourString('%s'%str(tensor.shape), Colours.OKGREEN) + ' ' + usrmsg
    else:
        msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': (' + colourString(str(tensor.dtype) + ' ' + str(type(tensor)), Colours.WARNING) + ') -- '  + colourString('%s'%str(tensor.shape), Colours.OKGREEN) + ' ' + usrmsg
    print(msg)
#
# Print debugging information.
def dprint(usrmsg, leveloffset=0):
    msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': ' + str(usrmsg)
    print(msg)

def hasNAN(t):
    msg = colourString(colourString(getLineInfo(), Colours.UNDERLINE), Colours.OKBLUE) + ': ' + colourString(str('Tensor has %s NaNs'%str((t != t).sum().item())), Colours.FAIL)
    print(msg)

def torch_mem():
    dprint('Torch report: Allocated: %.2f MBytes Cached: %.2f' % (torch.cuda.memory_allocated() / (1024 ** 2), torch.cuda.memory_cached() / (1024 ** 2)), 1)

## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('Torch report: %.2f MBytes' % (torch.cuda.memory_allocated() / (1024 ** 2)))
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)
