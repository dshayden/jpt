from abc import ABC, abstractmethod

class Serializable(ABC):
  _magicKey = '__jptSerializable'
  _classKey = '__jptClass'

  @abstractmethod
  def serializable(self): pass

  @abstractmethod
  def fromSerializable(): pass
