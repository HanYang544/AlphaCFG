from alphacfg.data.expression import Feature, Ref
from alphacfg.data.stock_data import FeatureType


high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
target = Ref(close, -20) / close - 1
