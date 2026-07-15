from typing import Iterator, Any

from evutils.io._source import make_source
from evutils.io.decoders import resolve_decoder_cls
from evutils.types import TriggerArray

class EventStreamer:
    """A low-level streamer that yields raw event blocks exactly as the parser outputs them.
    It performs ZERO slicing or logic; it purely drives the decoder.
    """
    def __init__(self, file: Any, ext_trigger: bool = False, decoder_cls: Any = None, **kwargs: Any):
        self.source = make_source(file)
        if decoder_cls is None:
            decoder_cls = resolve_decoder_cls(self.source)
        
        # Set the flag as an attribute after construction (as EventReader does)
        # rather than forcing it through every decoder's __init__: several
        # decoders (DAT/NPZ/HDF5/CSV) don't accept it as a constructor kwarg.
        self.decoder = decoder_cls(self.source, **kwargs)
        self.decoder.read_external_triggers = ext_trigger
        self.ext_trigger = ext_trigger

    def __iter__(self) -> Iterator[Any]:
        self.decoder.init()
        while True:
            chunk = self.decoder.read_chunk()
            
            if self.ext_trigger:
                if isinstance(chunk, tuple):
                    ev_chunk, tr_chunk = chunk
                else:
                    ev_chunk = chunk
                    tr_chunk = TriggerArray.empty()
            else:
                ev_chunk = chunk
                tr_chunk = None
                
            if len(ev_chunk) == 0 and (tr_chunk is None or len(tr_chunk) == 0):
                break
                
            if self.ext_trigger:
                yield ev_chunk, tr_chunk
            else:
                yield ev_chunk


