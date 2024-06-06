local get_time = require("debug.get_time")

local timers = {}

local function measure(what)
    if what then
        table.insert(timers, 1, {what = what, time = get_time()})
    else
        local t = table.remove(timers, 1)
        local time = get_time() - t.time
        print(string.format("%s took %.5f seconds", t.what, time))
        return time
    end
end

return measure