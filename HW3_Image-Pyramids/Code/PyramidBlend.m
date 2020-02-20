function output = PyramidBlend(source, mask, target)

output = source .* mask + target .* ~mask;