import math
from perceval import BS, PS, BasicState, RemoteProcessor
from perceval.algorithm import Sampler

proc = RemoteProcessor("sim:slos")
circuit = BS() // PS(phi=math.pi/4) // BS()
proc.set_circuit(circuit)

proc.with_input(BasicState([1, 1]))

proc.min_detected_photons_filter(2)

sampler = Sampler(proc, max_shots_per_call=10_000)
out = sampler.sample_count(5_000)
print(out["results"])