import os
import random
import hashlib
import argparse
from tqdm import tqdm

class SimpleCPUSimulator:
    def __init__(self,
                 num_registers=10,
                 memory_size=1024,
                 random_init=False,
                 pc=None,
                 acc=None,
                 ir=None,
                 registers=None):
        # Define the instruction set
        self.map_instruction = {
                                "HLT": 0,       # Halting instruction (zero-address instruction): "HLT" halt the CPU
                                "CLR": 1,       # Clear instruction (zero-address instruction, single-address instruction): "CLR" clear the accumulator; "CLR A" clear register A
                                "INC": 2,       # Increment instruction (zero-address instruction, single-address instruction): "INC" accumulator plus one; "INC A" register A plus one
                                "DEC": 3,       # Decrement instruction (zero-address instruction, single-address instruction): "DEC" accumulator minus one; "DEC A" register A minus one
                                "SHL": 4,       # Left shift instruction (zero-address instruction, single-address instruction): "SHL" accumulator left shift one bit; "SHL A" register A left shift one bit
                                "SHR": 5,       # Right shift instruction (zero-address instruction, single-address instruction): "SHR" accumulator right shift one bit; "SHR A" register A right shift one bit
                                "ROL": 6,       # Rotate left instruction (zero-address instruction, single-address instruction): "ROL" accumulator rotate left one bit; "ROL A" register A rotate left one bit
                                "ROR": 7,       # Rotate right instruction (zero-address instruction, single-address instruction): "ROR" accumulator rotate right one bit; "ROR A" register A rotate right one bit
                                "NOT": 8,       # Not instruction (zero-address instruction, single-address instruction): "NOT" accumulator not to accumulator; "NOT A" register A not to register A
                                "PUSH": 9,      # Push instruction (single-address instruction): "PUSH A" push the value of register A to the accumulator
                                "POP": 10,      # Pop instruction (single-address instruction): "POP A" pop the value of the accumulator to register A
                                "LOADI": 11,    # Immediate loading instruction (single-address instruction): "LOADI 1" load the immediate number 1 to the accumulator
                                "SWAP": 12,     # Swap instruction (single-address instruction, double-address instruction): "SWAP" swap the value of the accumulator and register A; "SWAP A B" swap the value of register A and register B
                                "ADD": 13,      # Addition instruction (single-address instruction, double-address instruction, three-address instruction): "ADD A" accumulator plus register A to accumulator; "ADD A B" register A plus register B to register A; "ADD A B C" register A plus register B to register C
                                "SUB": 14,      # Subtraction instruction (single-address instruction, double-address instruction, three-address instruction): "SUB A" accumulator minus register A to accumulator; "SUB A B" register A minus register B to register A; "SUB A B C" register A minus register B to register C
                                "MUL": 15,      # Multiplication instruction (single-address instruction, double-address instruction, three-address instruction): "MUL A" accumulator multiply register A to accumulator; "MUL A B" register A multiply register B to register A; "MUL A B C" register A multiply register B to register C
                                "DIV": 16,      # Division instruction (single-address instruction, double-address instruction, three-address instruction): "DIV A" accumulator divide register A to accumulator; "DIV A B" register A divide register B to register A; "DIV A B C" register A divide register B to register C
                                "AND": 17,      # And operation instruction (single-address instruction, double-address instruction, three-address instruction): "AND A" accumulator and register A to accumulator; "AND A B" register A and register B to register A; "AND A B C" register A and register B to register C
                                "OR": 18,       # Or operation instruction (single-address instruction, double-address instruction, three-address instruction): "OR A" accumulator or register A to accumulator; "OR A B" register A or register B to register A; "OR A B C" register A or register B to register C
                                "XOR": 19,      # Xor operation instruction (single-address instruction, double-address instruction, three-address instruction): "XOR A" accumulator xor register A to accumulator; "XOR A B" register A xor register B to register A; "XOR A B C" register A xor register B to register C
                                "MOV": 20,      # Move instruction (single-address instruction): "MOV A B" move the value of register B to register A
        }

        # Define memory, program counter, accumulator, and instruction register
        self.memory = [0] * memory_size
        self.PC = 0
        self.ACC = 0
        self.IR = [0] * 4

        # Define general-purpose registers
        self.registers = [0] * num_registers
        self.map_register = {}
        for i in range(num_registers):
            self.map_register[chr(ord('A') + i)] = i + 1
        
        # Randomly initialize the CPU
        if random_init:
            self.ACC = random.randint(0, 255)
            self.registers = [random.randint(0, 255) for _ in range(num_registers)]
        elif pc is not None and acc is not None and ir is not None and registers is not None:
            self.PC = pc
            self.ACC = acc
            self.IR = ir
            self.registers = registers

    def _fetch(self):
        # Fetch the instruction
        self.IR = self.memory[self.PC * 4: self.PC * 4 + 4]
        self.PC += 1
        self.PC = min(self.PC, len(self.memory) // 4 - 1)

    def _decode(self):
        # Decode the instruction
        op, addr1, addr2, addr3 = self.IR
        return op, addr1, addr2, addr3

    def _execute(self, op, addr1, addr2, addr3):
        old_acc = self.ACC
        old_registers = self.registers.copy()

        # Execute the instruction
        if op == self.map_instruction["HLT"]:
            # Halt instruction
            return False
        elif op == self.map_instruction["CLR"]:
            # Clear instruction
            if addr1 == 0:
                self.ACC = 0
            else:
                self.registers[addr1 - 1] = 0
        elif op == self.map_instruction["INC"]:
            # Increment instruction
            if addr1 == 0:
                self.ACC = min(self.ACC + 1, 255)
            else:
                self.registers[addr1 - 1] = min(self.registers[addr1 - 1] + 1, 255)
        elif op == self.map_instruction["DEC"]:
            # Decrement instruction
            if addr1 == 0:
                self.ACC = max(self.ACC - 1, 0)
            else:
                self.registers[addr1 - 1] = max(self.registers[addr1 - 1] - 1, 0)
        elif op == self.map_instruction["SHL"]:
            # Shift left instruction
            if addr1 == 0:
                self.ACC = (self.ACC << 1) & 0xFF
            else:
                self.registers[addr1 - 1] = (self.registers[addr1 - 1] << 1) & 0xFF
        elif op == self.map_instruction["SHR"]:
            # Shift right instruction
            if addr1 == 0:
                self.ACC = (self.ACC >> 1) & 0xFF
            else:
                self.registers[addr1 - 1] = (self.registers[addr1 - 1] >> 1) & 0xFF
        elif op == self.map_instruction["ROL"]:
            # Rotate left instruction
            if addr1 == 0:
                self.ACC = ((self.ACC << 1) & 0xFF) | ((self.ACC >> 7) & 0xFF)
            else:
                self.registers[addr1 - 1] = ((self.registers[addr1 - 1] << 1) & 0xFF) | ((self.registers[addr1 - 1] >> 7) & 0xFF)
        elif op == self.map_instruction["ROR"]:
            # Rotate right instruction
            if addr1 == 0:
                self.ACC = ((self.ACC >> 1) & 0xFF) | ((self.ACC << 7) & 0xFF)
            else:
                self.registers[addr1 - 1] = ((self.registers[addr1 - 1] >> 1) & 0xFF) | ((self.registers[addr1 - 1] << 7) & 0xFF)
        elif op == self.map_instruction["PUSH"]:
            # Push instruction
            self.ACC = self.registers[addr1 - 1]
        elif op == self.map_instruction["POP"]:
            # Pop instruction
            self.registers[addr1 - 1] = self.ACC
        elif op == self.map_instruction["NOT"]:
            # Not instruction
            if addr1 == 0:
                self.ACC = ~self.ACC & 0xFF
            else:
                self.registers[addr1 - 1] = ~self.registers[addr1 - 1] & 0xFF
        elif op == self.map_instruction["LOADI"]:
            # Immediate loading instruction
            self.ACC = addr1
        elif op == self.map_instruction["SWAP"]:
            # Swap instruction
            if addr2 == 0:
                self.ACC, self.registers[addr1 - 1] = self.registers[addr1 - 1], self.ACC
            else:
                self.registers[addr1 - 1], self.registers[addr2 - 1] = self.registers[addr2 - 1], self.registers[addr1 - 1]
        elif op == self.map_instruction["ADD"]:
            # Addition instruction
            if addr2 == 0:
                self.ACC = min(self.ACC + self.registers[addr1 - 1], 255)
            elif addr3 == 0:
                self.registers[addr1 - 1] = min(self.registers[addr1 - 1] + self.registers[addr2 - 1], 255)
            else:
                self.registers[addr3 - 1] = min(self.registers[addr1 - 1] + self.registers[addr2 - 1], 255)
        elif op == self.map_instruction["SUB"]:
            # Subtraction instruction
            if addr2 == 0:
                self.ACC = max(self.ACC - self.registers[addr1 - 1], 0)
            elif addr3 == 0:
                self.registers[addr1 - 1] = max(self.registers[addr1 - 1] - self.registers[addr2 - 1], 0)
            else:
                self.registers[addr3 - 1] = max(self.registers[addr1 - 1] - self.registers[addr2 - 1], 0)
        elif op == self.map_instruction["MUL"]:
            # Multiplication instruction
            if addr2 == 0:
                self.ACC = min(self.ACC * self.registers[addr1 - 1], 255)
            elif addr3 == 0:
                self.registers[addr1 - 1] = min(self.registers[addr1 - 1] * self.registers[addr2 - 1], 255)
            else:
                self.registers[addr3 - 1] = min(self.registers[addr1 - 1] * self.registers[addr2 - 1], 255)
        elif op == self.map_instruction["DIV"]:
            # Division instruction
            if addr2 == 0:
                if self.registers[addr1 - 1] == 0:
                    self.ACC = 255
                else:
                    self.ACC = self.ACC // self.registers[addr1 - 1]
            elif addr3 == 0:
                if self.registers[addr2 - 1] == 0:
                    self.registers[addr1 - 1] = 255
                else:
                    self.registers[addr1 - 1] = self.registers[addr1 - 1] // self.registers[addr2 - 1]
            else:
                if self.registers[addr2 - 1] == 0:
                    self.registers[addr3 - 1] = 255
                else:
                    self.registers[addr3 - 1] = self.registers[addr1 - 1] // self.registers[addr2 - 1]
        elif op == self.map_instruction["AND"]:
            # And operation instruction
            if addr2 == 0:
                self.ACC = (self.ACC & self.registers[addr1 - 1]) & 0xFF
            elif addr3 == 0:
                self.registers[addr1 - 1] = (self.registers[addr1 - 1] & self.registers[addr2 - 1]) & 0xFF
            else:
                self.registers[addr3 - 1] = (self.registers[addr1 - 1] & self.registers[addr2 - 1]) & 0xFF
        elif op == self.map_instruction["OR"]:
            # Or operation instruction
            if addr2 == 0:
                self.ACC = (self.ACC | self.registers[addr1 - 1]) & 0xFF
            elif addr3 == 0:
                self.registers[addr1 - 1] = (self.registers[addr1 - 1] | self.registers[addr2 - 1]) & 0xFF
            else:
                self.registers[addr3 - 1] = (self.registers[addr1 - 1] | self.registers[addr2 - 1]) & 0xFF
        elif op == self.map_instruction["XOR"]:
            # Xor operation instruction
            if addr2 == 0:
                self.ACC = (self.ACC ^ self.registers[addr1 - 1]) & 0xFF
            elif addr3 == 0:
                self.registers[addr1 - 1] = (self.registers[addr1 - 1] ^ self.registers[addr2 - 1]) & 0xFF
            else:
                self.registers[addr3 - 1] = (self.registers[addr1 - 1] ^ self.registers[addr2 - 1]) & 0xFF
        elif op == self.map_instruction["MOV"]:
            # Move instruction
            self.registers[addr1 - 1] = self.registers[addr2 - 1]
        else:
            raise Exception("Invalid instruction")
        
        new_acc = self.ACC
        new_registers = self.registers.copy()

        for _ in [new_acc] + new_registers:
            if _ > 255 or _ < 0:
                print("Invalid instruction: %s" % self.back_translate(self.IR))
                print("ACC: %d -> %d" % (old_acc, new_acc))
                print("Registers: %s -> %s" % (old_registers, new_registers))

        return True

    def run(self, states_path):
        # Run the CPU and save the memory and CPU states
        with open(states_path, 'wb') as f:
            f.write(self.export_state())

        while True:
            self._fetch()
            op, addr1, addr2, addr3 = self._decode()
            # After each instruction, save the memory and CPU states
            # Skip saving the memory as it is not changed
            with open(states_path, 'ab') as f:
                f.write(self.export_state()[len(self.memory):])

            if not self._execute(op, addr1, addr2, addr3):
                break

    def export_state(self):
        # Export the CPU state as a byte stream
        state = []
        state += self.memory
        state += [self.PC, self.ACC]
        state += self.IR
        state += self.registers

        return bytes(state)
    
    def load_last_state(self, states_path):
        # Load the last CPU state from the byte stream
        with open(states_path, 'rb') as f:
            states = f.read()

        states = bytes(states)

        if len(states) < 6 + len(self.registers) + len(self.memory):
            raise Exception("Invalid states file")
        
        # Load the last CPU state
        self.memory = states[:len(self.memory)]
        last_state = states[-(6 + len(self.registers)):]

        self.PC = last_state[0]
        self.ACC = last_state[1]
        self.IR = last_state[2:6]
        for i in range(len(self.registers)):
            self.registers[i] = last_state[6 + i]

    def load_program(self, program):
        # Load the program into the memory
        program = [self.translate(*instruction.split(" ")) for instruction in program]
        program = [byte for instruction in program for byte in instruction]
        self.memory = program + self.memory[len(program):]
    
    def back_translate_program(self, program):
        # Translate the program from machine instructions to assembly instructions
        program = [self.back_translate(program[i:i+4]) for i in range(0, len(program), 4)]
        # Only keep one "HLT" instruction
        if "HLT" in program:
            program = program[:program.index("HLT") + 1]
        return program

    def translate(self, op, addr1="", addr2="", addr3=""):
        # Translate the assembly instruction to a machine instruction
        op = self.map_instruction[op]

        if addr1 != "" and op != self.map_instruction["LOADI"]:
            addr1 = self.map_register[addr1]
        elif op == self.map_instruction["LOADI"]:
            addr1 = int(addr1)
        else:
            addr1 = 0

        if addr2 != "":
            addr2 = self.map_register[addr2]
        else:
            addr2 = 0

        if addr3 != "":
            addr3 = self.map_register[addr3]
        else:
            addr3 = 0

        return [op, addr1, addr2, addr3]
    
    def back_translate(self, instruction):
        # Translate the machine instruction to an assembly instruction
        op, addr1, addr2, addr3 = instruction
        op = list(self.map_instruction.keys())[list(self.map_instruction.values()).index(op)]

        if addr1 != 0 and op != "LOADI":
            addr1 = list(self.map_register.keys())[list(self.map_register.values()).index(addr1)]
        elif op == "LOADI":
            addr1 = str(addr1)
        else:
            addr1 = ""

        if addr2 != 0:
            addr2 = list(self.map_register.keys())[list(self.map_register.values()).index(addr2)]
        else:
            addr2 = ""

        if addr3 != 0:
            addr3 = list(self.map_register.keys())[list(self.map_register.values()).index(addr3)]
        else:
            addr3 = ""

        return " ".join([op, addr1, addr2, addr3]).strip()
    
    def translate_states(self, states_path):
        # Translate the byte stream to the memory and CPU states
        with open(states_path, 'rb') as f:
            states = f.read()

        result = ""

        # Split the byte stream into memory and CPU states
        states = list(states)
        memory = states[:len(self.memory)]
        states = states[len(memory):]

        print("Program:", self.back_translate_program(memory))
        print()
        result += "Program: " + str(self.back_translate_program(memory)) + "\n\n"

        # Split the CPU states into individual states
        for i in range(0, len(states), 6 + len(self.registers)):
            PC = states[i]
            ACC = states[i + 1]
            IR = states[i + 2: i + 6]
            registers = {}
            for j in range(len(self.registers)):
                registers[chr(ord('A') + j)] = states[i + 6 + j]

            print("State at step %d:" % (i // (6 + len(self.registers))))
            print("PC:", PC)
            print("ACC:", ACC)
            print("IR:", self.back_translate(IR))
            print("Registers:", registers)
            print()
            result += "State at step %d:\n" % (i // (6 + len(self.registers)))
            result += "PC: " + str(PC) + "\n"
            result += "ACC: " + str(ACC) + "\n"
            result += "IR: " + str(self.back_translate(IR)) + "\n"
            result += "Registers: " + str(registers) + "\n\n"
        
        return result

    def random_program(self, num_instructions=10):
        # Generate a random program
        # Define the instruction set based on address number
        addr_num_instruction = {
                                    "CLR": [0, 1],      
                                    "INC": [0, 1],      
                                    "DEC": [0, 1],      
                                    "SHL": [0, 1],      
                                    "SHR": [0, 1],      
                                    "ROL": [0, 1],      
                                    "ROR": [0, 1],      
                                    "NOT": [0, 1],      
                                    "PUSH": [1],        
                                    "POP": [1],         
                                    "LOADI": [1],       
                                    "SWAP": [1, 2],     
                                    "ADD": [1, 2, 3],   
                                    "SUB": [1, 2, 3],   
                                    "MUL": [1, 2, 3],   
                                    "DIV": [1, 2, 3],   
                                    "AND": [1, 2, 3],   
                                    "OR": [1, 2, 3],    
                                    "XOR": [1, 2, 3],   
                                    "MOV": [2],         
            }
        
        # Generate a random program
        program = []

        for i in range(num_instructions-1):
            instruction = random.choice(list(addr_num_instruction.keys()))
            if instruction == "LOADI":
                program.append(instruction + " " + str(random.randint(0, 255)))
            else:
                addr_num = random.choice(addr_num_instruction[instruction])
                if addr_num == 0:
                    program.append(instruction)
                elif addr_num == 1:
                    program.append(instruction + " " + random.choice(list(self.map_register.keys())))
                elif addr_num == 2:
                    program.append(instruction + " " + random.choice(list(self.map_register.keys())) + " " + random.choice(list(self.map_register.keys())))
                else:
                    program.append(instruction + " " + random.choice(list(self.map_register.keys())) + " " + random.choice(list(self.map_register.keys())) + " " + random.choice(list(self.map_register.keys())))

        program.append("HLT")

        return program

def calculate_md5_of_list(input_list):
    # Convert list to string
    list_str = ''.join(str(e) for e in input_list)
    md5_hash = hashlib.md5(list_str.encode()).hexdigest()

    return md5_hash

def argument_parser():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="CPU Simulation")
    parser.add_argument("--mode", type=str, default="generate", help="Mode of the CPU simulation. Options: 'generate', 'translate', and 'evaluate'")
    parser.add_argument("--num_registers", type=int, default=10, help="Number of registers in the CPU")
    parser.add_argument("--memory_size", type=int, default=1024, help="Size of the memory in the CPU")
    parser.add_argument("--dir_path", type=str, default="cpu_states", help="Path to generate or evaluate the CPU states, only for 'generate' and 'evaluate' modes")
    parser.add_argument("--states_path", type=str, default="cpu_states/cpu.bin", help="Path to the CPU states, only for 'translate' mode")
    parser.add_argument("--num_min_program", type=int, default=1, help="Minimum number of instructions in the program, only for 'generate' mode")
    parser.add_argument("--num_max_program", type=int, default=255, help="Maximum number of instructions in the program, only for 'generate' mode")
    parser.add_argument("--num_train_instance", type=int, default=2100000, help="Number of training instances, only for 'generate' mode")
    parser.add_argument("--num_test_instance", type=int, default=21000, help="Number of testing instances, only for 'generate' mode")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_path", type=str, default="cpu_states.txt", help="Path to save the translated CPU states, only for 'translate' mode")

    return parser.parse_args()

if __name__ == "__main__":
    args = argument_parser()
    mode = args.mode
    num_registers = args.num_registers
    memory_size = args.memory_size
    dir_path = args.dir_path
    states_path = args.states_path
    num_min_program = args.num_min_program
    num_max_program = args.num_max_program
    num_train_instance = args.num_train_instance
    num_test_instance = args.num_test_instance
    random_seed = args.random_seed
    output_path = args.output_path

    if mode == "generate":
        random.seed(random_seed)
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(dir_path+"/train", exist_ok=True)
        os.makedirs(dir_path+"/test", exist_ok=True)
        print("Generating the CPU States dataset with %d training instances and %d testing instances at %s" % (num_train_instance, num_test_instance, dir_path))

        for _ in tqdm(range(num_train_instance)):
            cpu = SimpleCPUSimulator(num_registers=num_registers,
                                     memory_size=memory_size,
                                     random_init=True)

            num_program = random.randint(num_min_program, num_max_program)
            program = cpu.random_program(num_program)
            filename = dir_path+"/train/"+calculate_md5_of_list(program) + ".bin"

            cpu.load_program(program)
            cpu.run(filename)

        for _ in tqdm(range(num_test_instance)):
            cpu = SimpleCPUSimulator(num_registers=num_registers,
                                    memory_size=memory_size,
                                    random_init=True)

            num_program = random.randint(num_min_program, num_max_program)
            program = cpu.random_program(num_program)
            filename = dir_path+"/test/"+calculate_md5_of_list(program) + ".bin"

            cpu.load_program(program)
            cpu.run(filename)

    elif mode == "translate":
        print("Translating the CPU states at %s" % states_path)
        cpu = SimpleCPUSimulator(num_registers=num_registers,
                                 memory_size=memory_size)
        result = cpu.translate_states(states_path)
        with open(output_path, 'w') as f:
            f.write(result)

    elif mode == "evaluate":
        matched_bytes = 0
        total_bytes = 0
        eval_set = tqdm(os.listdir(dir_path))
        print("Evaluating the accuracy of the CPU simulation")

        for states_path in eval_set:
            with open(dir_path+"/"+states_path, 'rb') as f:
                states = f.read()
            
            states = list(states)
            program = states[:memory_size]
            first_state = states[memory_size:memory_size+6+num_registers]

            cpu = SimpleCPUSimulator(num_registers=num_registers,
                                     memory_size=memory_size,
                                     random_init=False,
                                     pc=first_state[0],
                                     acc=first_state[1],
                                     ir=first_state[2:6],
                                     registers=first_state[6:])
            program = cpu.back_translate_program(program)
            cpu.load_program(program)
            cpu.run("temp.bin")

            with open("temp.bin", 'rb') as f:
                ground_truth_states = f.read()
            
            ground_truth_states = list(ground_truth_states)
            
            for i in range(memory_size+6+num_registers, len(ground_truth_states)):
                total_bytes += 1

                if len(states) <= i:
                    total_bytes += len(ground_truth_states) - i
                    break

                if states[i] == ground_truth_states[i]:
                    matched_bytes += 1

            if total_bytes != matched_bytes:
                cpu.translate_states("temp.bin")
                cpu.translate_states(dir_path+"/"+states_path)
                break
            eval_set.set_postfix({"Accuracy": matched_bytes/total_bytes})
        
        os.remove("temp.bin")
