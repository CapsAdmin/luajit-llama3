local ffi = require 'ffi'

-- Define necessary C types and functions
ffi.cdef[[
typedef struct { uint8_t b[4096]; } code_t;  // 4KB should be enough for our example
void* mmap(void* addr, size_t len, int prot, int flags, int fd, size_t off);
int mprotect(void* addr, size_t len, int prot);
int munmap(void* addr, size_t length);

// Constants for mmap
static const int PROT_READ = 1;
static const int PROT_WRITE = 2;
static const int PROT_EXEC = 4;
static const int MAP_PRIVATE = 2;
static const int MAP_ANONYMOUS = 32;
]]

-- Simple x64 assembler
local Assembler = {}
Assembler.__index = Assembler

function Assembler.new()
    local self = setmetatable({
        code = {},
        pos = 1
    }, Assembler)
    return self
end

-- Helper to emit bytes
function Assembler:emit(...)
    for i, b in ipairs({...}) do
        self.code[self.pos] = b
        self.pos = self.pos + 1
    end
end

-- REX prefix for 64-bit operations
function Assembler:rex(w, r, x, b)
    local rex = 0x40
    if w then rex = rex + 8 end
    if r then rex = rex + 4 end
    if x then rex = rex + 2 end
    if b then rex = rex + 1 end
    self:emit(rex)
end

-- ModRM byte
function Assembler:modrm(mod, reg, rm)
    self:emit(bit.bor(bit.lshift(mod, 6), bit.lshift(reg, 3), rm))
end

-- Basic x64 instructions
function Assembler:mov_rax_imm64(imm)
    self:rex(true)
    self:emit(0xB8)  -- MOV RAX, imm64
    self:emit_u64(imm)
end

function Assembler:mov_rcx_imm64(imm)
    self:rex(true)
    self:emit(0xB9)  -- MOV RCX, imm64
    self:emit_u64(imm)
end

function Assembler:vmovups_ymm0_mem(base_reg)
    self:emit(0xC5, 0xFC, 0x10, 0x00 + base_reg)  -- VMOVUPS ymm0, [base_reg]
end

function Assembler:vmovups_ymm1_mem(base_reg)
    self:emit(0xC5, 0xFC, 0x10, 0x08 + base_reg)  -- VMOVUPS ymm1, [base_reg]
end

function Assembler:vfmadd231ps_ymm0_ymm1(base_reg)
    self:emit(0xC4, 0xE2, 0x75, 0xB8, 0x00 + base_reg)  -- VFMADD231PS ymm0, ymm1, [base_reg]
end

function Assembler:vmovups_mem_ymm0(base_reg)
    self:emit(0xC5, 0xFC, 0x11, 0x00 + base_reg)  -- VMOVUPS [base_reg], ymm0
end

function Assembler:ret()
    self:emit(0xC3)
end

-- Helper to emit 64-bit integer
function Assembler:emit_u64(n)
    for i = 0, 7 do
        self:emit(bit.band(bit.rshift(n, i*8), 0xFF))
    end
end

-- Build the machine code
function Assembler:build()
    -- Allocate executable memory
    local size = 4096  -- One page
    local prot = bit.bor(ffi.C.PROT_READ, ffi.C.PROT_WRITE, ffi.C.PROT_EXEC)
    local flags = bit.bor(ffi.C.MAP_PRIVATE, ffi.C.MAP_ANONYMOUS)
    local ptr = ffi.C.mmap(nil, size, prot, flags, -1, 0)
    
    -- Copy machine code to executable memory
    local code_ptr = ffi.cast("uint8_t*", ptr)
    for i = 1, #self.code do
        code_ptr[i-1] = self.code[i]
    end
    
    -- Cast to function pointer
    return ffi.cast("void(*)(void*, void*, void*)", ptr)
end

-- Example: Create a function that performs SIMD matrix-vector multiplication
local function create_matmul_func()
    local asm = Assembler.new()
    
    -- Function takes three parameters in RDI (matrix), RSI (vector), RDX (result)
    -- Example for 8 floats (one AVX register)
    
    -- Load vector elements into YMM1
    asm:vmovups_ymm1_mem(6)  -- [RSI]
    
    -- Load matrix row into YMM0
    asm:vmovups_ymm0_mem(7)  -- [RDI]
    
    -- Multiply and accumulate
    asm:vfmadd231ps_ymm0_ymm1(7)
    
    -- Store result
    asm:vmovups_mem_ymm0(2)  -- [RDX]
    
    -- Return
    asm:ret()
    
    return asm:build()
end

-- Test the assembly function
local function test_matmul()
    -- Allocate aligned memory for matrix, vector, and result
    local matrix = ffi.new("float[8][8]")
    local vector = ffi.new("float[8]")
    local result = ffi.new("float[8]")
    
    -- Initialize test data
    for i = 0, 7 do
        vector[i] = i + 1
        for j = 0, 7 do
            matrix[i][j] = i + j + 1
        end
    end
    
    -- Create and call the function
    local func = create_matmul_func()
    func(matrix, vector, result)
    
    -- Print results
    print("Result:")
    for i = 0, 7 do
        print(result[i])
    end
end

-- Run the test
test_matmul()