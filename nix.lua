package.path = './?.lua;' .. package.path
local args = {...}
assert(loadfile(args[1]))(unpack(args, 2))