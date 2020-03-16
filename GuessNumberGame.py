a = 1
b = 1000
print(f'Загадай число от {a} до {b}...')

# Даем время на подумать
import time
time.sleep(1)
print('Загадал?')
time.sleep(1)
print('Тогда поехали!')
print()
time.sleep(1)
rules = '''Если я угадаю, напиши "=",
если твое число меньше, то введи "<",
а если больше, то ">".
И нажми на Enter.
'''
print(rules)
time.sleep(1)

# Отгадываем
steps = 0

while True:
    if a > b:
        print('Произошел конфуз...')
        print('Попробуй заново')
        break
    if a == b:
        print(f'Все понятно. Ответ - {a}')
        print(f'Потребовалось шагов: {steps}')
        break

    steps += 1

    number = (a + b) // 2
    answer = input(f'Это {number}? ')

    if answer == '=':
        print('Ура! Я отгадал!')
        print(f'Потребовалось шагов: {steps}')
        break
    elif answer == '<':
        b = number - 1
    elif answer == '>':
        a = number + 1
    else:
        print(rules)
        steps -= 1