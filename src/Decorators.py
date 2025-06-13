import logging
import time


# 记录函数耗时的装饰器
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        logging.info(f"开始执行{func.__name__} 任务.")
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"任务 {func.__name__} 执行完毕. 总耗时为: {end_time - start_time} 秒.")
        return output
    return wrapper
