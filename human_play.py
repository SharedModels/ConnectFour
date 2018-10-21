class HumanPlay:
    def action(self, state, c, oc):
        print(state)
        correct = True
        while correct:
            print('Type attack:')
            attack = input()
            attack = int(attack)
            if attack > 6 | attack < 0:
                print('Attack must be between 0 and 6')
            else:
                return attack