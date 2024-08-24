import queue
import threading
class ObjectPool:
    def __init__(self, factory_func, max_size = 5):
        self.pool = queue.Queue(maxsize=max_size)
        for _ in range(max_size):
            self.pool.put(factory_func())

    def __len__(self):
        return self.pool.qsize()
    def get(self, timeout=None):
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("object_pool_timeout")
    def put(self, obj):
        self.pool.put(obj)