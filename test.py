from random import randint

import simpy


#
# class EV:
#     def __init__(self, env):
#         self.env = env
#         self.drive_proc = env.process(self.drive(env))
#
#     def drive(self, env):
#         while True:
#             # Drive for 20-40 min
#             yield env.timeout(randint(20, 40))
#
#             # Park for 1 hour
#             print('Start parking at', env.now)
#             charging = env.process(self.bat_ctrl(env))
#             parking = env.timeout(60)
#             yield charging | parking
#             if not charging.triggered:
#                 # Interrupt charging if not already done.
#                 charging.interrupt('Need to go!')
#             print('Stop parking at', env.now)
#             yield env.timeout(10)
#             yield charging
#             print("done charging at", env.now)
#
#     def bat_ctrl(self, env):
#         print('Bat. ctrl. started at', env.now)
#         try:
#             yield env.timeout(randint(60, 90))
#             print('Bat. ctrl. done at', env.now)
#         except simpy.Interrupt as i:
#             # Onoes! Got interrupted before the charging was done.
#             print('Bat. ctrl. interrupted at', env.now, 'msg:',
#                   i.cause)


class PCOAgent1:
    def __init__(self, env):

        self.env = env

        # initialise 1000 s timer
        # self.timer = self.env.timeout(1000, value='long')

        self.timeCounter = 0
        self.period = 1000
        env.process(self.main())

    # def neighbors_in_range(dist):
    #     pass

    def main(self):
        run = self.env.process(self.run())
        timeout = self.env.timeout(5)

        yield run | timeout
        if not run.triggered:
            run.interrupt('message received')

    # called (once) automatically when the agent is created
    def run(self):
        not_done = True
        period = self.env.timeout(self.period)
        while not_done:
            # long timer wakeup
            try:
                yield period
                not_done = False
                print('finished period timer', ' at', self.env.now)
            # message received interrupt
            except simpy.Interrupt as interrupt:
                # print("interrupt cause: ", interrupt.cause)
                if interrupt.cause == 'message received':
                    print('message received', ' at', self.env.now)
                else:
                    print('timer expired', ' at', self.env.now)


environment = simpy.Environment()
ev = PCOAgent1(environment)
# ev = EV(env)
# PCOAgent
environment.run(until=1100)
