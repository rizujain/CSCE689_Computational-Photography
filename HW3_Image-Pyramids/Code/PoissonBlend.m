function output = PoissonBlend(source, mask, target, isMix)

output = source .* mask + target .* ~mask;