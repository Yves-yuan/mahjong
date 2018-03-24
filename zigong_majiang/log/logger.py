import logging


class Logger(object):
    @staticmethod
    def init():
        # 创建一个logger
        logger = logging.getLogger('mahjong')
        logger.setLevel(logging.INFO)

        # 创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # 给logger添加handler
        logger.addHandler(ch)

        # 记录一条日志
        logger.info('init')
