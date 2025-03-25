import threading
import typing
from abc import ABC, abstractmethod

import numpy as np


class SignalProcessorBase(ABC):

    @abstractmethod
    def process(self, signal: np.ndarray, *args: typing.Any, **kwargs: typing.Any) -> np.ndarray:
        pass

    @staticmethod
    def get_label() -> str:
        pass

    @abstractmethod
    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)


class SignalReaderBase(ABC):

    @abstractmethod
    def read(self, signal: np.ndarray, *args: typing.Any, **kwargs: typing.Any):
        pass

    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)


class SignalProducerBase(ABC):
    def __init__(self):
        self.readers: [SignalReaderBase] = []
        self.running = False
        self.stopped = False

    def register_reader(self, reader: SignalReaderBase):
        self.readers.append(reader)

    def unregister_reader(self, reader: SignalReaderBase):
        self.readers.remove(reader)

    def clear_readers(self):
        self.readers.clear()

    def send_signal(self, signal, *args, **kwargs):
        # print("send to"+str(self.readers))
        for reader in self.readers:
            reader(signal, *args, **kwargs)

    @abstractmethod
    def _produce(self):
        pass

    @staticmethod
    def get_label():
        pass

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._produce).start()

    def stop(self):
        self.running = False

    def is_stopped(self):
        return self.stopped
    
    def __del__(self):
        self.stop()
