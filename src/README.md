# Example Python Script

# Results from Task 4:
FOL AGENT:
True state (hidden from the agent):
  1 2 3 4
4 . . . .
3 F P D .
2 . . . .
1 > . D .

Start at (1,1) facing EAST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Known safe: [(1, 1), (1, 2), (2, 1)]

Step 1: FORWARD
  Position: (2,1), Facing: EAST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1)]
  Known dangerous: []

Step 2: TURN_RIGHT
  Position: (2,1), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 3: TURN_RIGHT
  Position: (2,1), Facing: WEST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 4: FORWARD
  Position: (1,1), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1)]
  Known dangerous: []

Step 5: TURN_RIGHT
  Position: (1,1), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 6: FORWARD
  Position: (1,2), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 7: TURN_RIGHT
  Position: (1,2), Facing: EAST
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 8: FORWARD
  Position: (2,2), Facing: EAST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 9: FORWARD
  Position: (3,2), Facing: EAST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 10: TURN_RIGHT
  Position: (3,2), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 11: TURN_RIGHT
  Position: (3,2), Facing: WEST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 12: FORWARD
  Position: (2,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 13: TURN_RIGHT
  Position: (2,2), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 14: FORWARD
  Position: (2,3), Facing: NORTH
  Percept: Percept(creaking=True, rumbling=True, beacon=True, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 15: GRAB
  Position: (2,3), Facing: NORTH
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'GRAB', 'grabbed': True}

Step 16: TURN_RIGHT
  Position: (2,3), Facing: EAST
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 17: TURN_RIGHT
  Position: (2,3), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 18: FORWARD
  Position: (2,2), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 19: TURN_RIGHT
  Position: (2,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 20: FORWARD
  Position: (1,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 21: TURN_LEFT
  Position: (1,2), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_LEFT'}

Step 22: FORWARD
  Position: (1,1), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 23: EXIT
  Position: (1,1), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'EXIT', 'exit': 'success'}

========================================
Episode ended.  Reward: 978
Steps taken: 23
Success: True


Propositional Agent
True state (hidden from the agent):
  1 2 3 4
4 . . . .
3 F P D .
2 . . . .
1 > . D .

Start at (1,1) facing EAST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Known safe: [(1, 1), (1, 2), (2, 1)]

Step 1: FORWARD
  Position: (2,1), Facing: EAST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1)]
  Known dangerous: []

Step 2: TURN_RIGHT
  Position: (2,1), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 3: TURN_RIGHT
  Position: (2,1), Facing: WEST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 4: FORWARD
  Position: (1,1), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1)]
  Known dangerous: []

Step 5: TURN_RIGHT
  Position: (1,1), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 6: FORWARD
  Position: (1,2), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 7: TURN_RIGHT
  Position: (1,2), Facing: EAST
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 8: FORWARD
  Position: (2,2), Facing: EAST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 9: FORWARD
  Position: (3,2), Facing: EAST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 10: TURN_RIGHT
  Position: (3,2), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 11: TURN_RIGHT
  Position: (3,2), Facing: WEST
  Percept: Percept(creaking=True, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 12: FORWARD
  Position: (2,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 13: TURN_RIGHT
  Position: (2,2), Facing: NORTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 14: FORWARD
  Position: (2,3), Facing: NORTH
  Percept: Percept(creaking=True, rumbling=True, beacon=True, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 15: GRAB
  Position: (2,3), Facing: NORTH
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'GRAB', 'grabbed': True}

Step 16: TURN_RIGHT
  Position: (2,3), Facing: EAST
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 17: TURN_RIGHT
  Position: (2,3), Facing: SOUTH
  Percept: Percept(creaking=True, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 18: FORWARD
  Position: (2,2), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 19: TURN_RIGHT
  Position: (2,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_RIGHT'}

Step 20: FORWARD
  Position: (1,2), Facing: WEST
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 21: TURN_LEFT
  Position: (1,2), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=True, beacon=False, bump=False, beep=False)
  Info: {'action': 'TURN_LEFT'}

Step 22: FORWARD
  Position: (1,1), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'FORWARD'}
  Known safe: [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
  Known dangerous: [(1, 3), (3, 1)]

Step 23: EXIT
  Position: (1,1), Facing: SOUTH
  Percept: Percept(creaking=False, rumbling=False, beacon=False, bump=False, beep=False)
  Info: {'action': 'EXIT', 'exit': 'success'}

========================================
Episode ended.  Reward: 978
Steps taken: 23
Success: True

Both implementations take the same amount of steps and receive the same reward

# Task 6