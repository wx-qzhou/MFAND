import codecs
import json
from os.path import join
import pickle
import os

# loads是将str转化成dict格式。
def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)

# dumps是将dict转化成str格式
def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


# 向文件中写入数据
def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)

# 从文件中读取数据
def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

# string 转 json
def serialize_embedding(embedding):
    return pickle.dumps(embedding)

# json 转 string
def deserialize_embedding(s):
    return pickle.loads(s)

# https://zhuanlan.zhihu.com/p/37534850
# 单例模式保证了在程序的不同位置都可以且仅可以取到同一个对象实例：
# 如果实例不存在，会创建一个实例；如果已存在就会返回这个实例。
# 因为单例是一个类，所以你也可以为其提供相应的操作方法，以便于对这个实例进行管理。
class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """
    """
    一个非线程的安全的助手类用于实现singletons。这个被用来作为一个decorator，这个不是一个metaclass（元类），而是能够实现singleton。
    这个修饰类定义一个“__init__”函数，只实现“self”参数，同时修饰类不能被继承，除此之外，对于修饰类就没有什么限制了。
    使用“Instance”方法来获取singleton实例。若是使用“__call__”将会导致“TypeError”。
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        """
        返回一个singleton实例，

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
