BOT_CONFIG = {
    'threshold': 0.6,

    'intents': {

        'hello': {
            'examples': ['Привет', 'Добрый день', 'Здраствуйте!', 'Шалом', 'Хай', 'Здорово', 'Хеллоу', 'Салют',
                         'Салам'],
            'responses': ['Привет, юзер!', 'Здраствуй', 'Хэй', 'Привет, пользователь']
        },
        'goodbye': {
            'examples': ['Пока', 'Всего доброго', 'До свидания', 'Бай бай', 'До связи', 'пока', 'досвидания', 'бб', 'давай пока', 'до встречи', 'увидимся'],
            'responses': ['Пока', 'Счастливо!', 'До завтра', 'Всего хорошего', 'Досвидания', 'Ещё увидимися!', 'До новых встреч', 'ББ', 'Пока, друг']
        },
        'thanks': {
            'examples': ['Спасибо', 'Спасибо большое!', 'Сенкс', 'Благодарю', 'Очень благодарен'],
            'responses': ['Вам спасибо!', 'Не за что', 'Нет проблем', 'Всегда пожалуйста']
        },
        'whatcanyoudo': {
            'examples': ['Что ты умеешь?', 'Расскажи что умеешь', 'Какие у тебя навыки?', 'Раскажи о своих навыках'],
            'responses': ['Отвечать на вопросы. Просто напиши :)', 'Могу ответить на твои вопросы']
        },
        'name': {
            'examples': ['Как тебя зовут?', 'Твое имя?', 'У тебя есть имя?'],
            'responses': ['Меня зовут Бот. Просто Бот.', 'Называй меня Бот', 'Мое имя Бот']
        },
        'weather': {
            'examples': ['Какая погода в Москве?', 'Какая погода?', 'Раскажи мне погоду'],
            'responses': ['Погода так себе...', 'Погода просто класс', 'Погода неочень. Лучше останься дома']
        },

        'meaning': {
            'examples': ['В чем смысл жизни?', 'В чем сущность бытия?'],
            'responses': ['В коде, друг!', 'В знаниях!', 'В Боге!']
        },
        'love': {
            'examples': ['Что ты любишь?', 'Что ты любишь делать?', 'Чем ты любишь заниматься?'],
            'responses': ['Я люблю отвечать на вопросы, и вести беседу с тобоюб друг!']
        },
        'father': {
            'examples': ['Кто тебя создал?', 'Кто твой папа?', 'Кто тебя придумал?'],
            'responses': ['Надо мной трудилась целая группа ученых', 'Питон - моя семья']
        },
        'coronavirus': {
            'examples': ['Как не заразиться коронавирусом?', 'Как спастись от коронавируса?',
                         'Как избежать коронавируса', 'Как не заболеть коронавирусом?'],
            'responses': ['Мойте руки', 'Сидите дома', 'Избегайте контакта с людьми']
        },
        'kids': {
            'examples': ['Ты любишь детей?', 'Тебе нравятся дети?'],
            'responses': ['Дети - цветы жизни!', 'Я люблю любить детей :)']
        },

        'pushkin': {
            'examples': ['А Пушкин?', 'Расскажи стих Пушкина?', 'Знаешь Пушкина?'],
            'responses': [
                'Я помню чудное мгновенье: Передо мной явилась ты, Как мимолетное виденье, Как гений чистой красоты.']
        },

        'lermontov': {
            'examples': ['А Лермонтов?', 'Расскажи стих Лермонтова?', 'Знаешь Лермонтова?'],
            'responses': [
                'Белеет парус одинокий, В тумане моря голубом! Что ищет он в стране далекой? Что кинул он в краю родном?']
        },
        'year': {
            'examples': ['какой сейчас год?', 'какой год?'],
            'responses': ['Сейчас 2020 год', 'Скорее всего 2020 год', 'Вчера еще был 2020 ;)']
        },
        'sense': {
            'examples': ['в чем смысл жизни?', 'в чем смысл?'],
            'responses': ['Я тоже об этом думал...', 'Смысл в ...<БОТ ВЫШЕЛ ИЗ СТРОЯ>', '42']
        },
        'age': {
            'examples': ['Сколько тебе лет?'],
            'responses': ['Я очень молод.', 'Это тайна)', 'Меня создали буквально вчера.']
        },
        'wherefrom': {
            'examples': ['Откуда ты?', 'Из какого ты города?', 'Из какой ты страны?'],
            'responses': ['Я из планеты Земля', 'Где-то из Млечного Пути', ]

        },

        'wheretogo': {
            'examples': ['Что сегодня идёт в кино?', 'Куда можно сводить девушку?', 'Где активно провести время?'],
            'responses': ['Минутку, подгружаю афишу']
        },
        'howareyou': {
            'examples': ['Как дела?', 'Как поживаешь?', 'Как настроение?'],
            'responses': ['Отлично, а у Вас?', 'У меня всё супер!', 'Всё хорошо, а как ваши дела?']
        },
        'currency': {
            'examples': ['Какой курс валюты сейчас?', 'Какой курс доллара?', 'Какой курс евро?'],
            'responses': ['Подождите, минутку, я сейчас уточню', 'курс доллара 72,62 но он пока захардкожен',
                          'Лучше не спрашивайте', 'Больше, чем вчера', 'Вы хотите испортить себе настроение?',
                          'Около 30 рублей!... Шутка!']
        },
        'hobby': {
            'examples': ['Какое у тебя хобби?', 'Чем ты занимаешься в всвободное время?'],
            'responses': ['О, я коллекционирую вопросы', 'Я люблю бегать на лыжах в свободное от работы время']
        },
        'wake': {
            'examples': ['Поставь будильник на 8 утра', 'Разбуди меня через 8 часов'],
            'responses': ['Конечно, будет сделано', 'Сделано']
        },

        'boss': {
            'examples': ['кто твой босс', 'кто твой начальник', 'кто ты думаешь твой начальник', 'кто твой хозяин',
                         'кто твой владелец', 'кто здесь главный', 'на кого ты работаешь'],
            'responses': ['Конечно ты', 'Это ты', 'Ты!']
        },

        'busy': {
            'examples': ['ты занят', 'тебе нужно многое сделать', 'ты очень занят да', 'ты работаешь',
                         'насколько ты занят', 'ты всё ещё работаешь над этим', 'ты работаешь сейчас'],
            'responses': ['Дел много, кручусь.', 'Стараюсь не сидеть на месте.']
        },

        'brith_date': {
            'examples': ['когда ты родился', 'твоя дата рождения', 'когда твой день рождения',
                         'когда ты празднуешь свой день рождения', 'когда у тебя день рождения', 'дата твоего рождения',
                         'когда ты родился', 'твой день рождения'],
            'responses': [
                'У меня нет какого-то конкретного дня рождения, поэтому я принимаю подарки и поздравления круглый год.',
                'А я даже не знаю, когда у меня день рождения. Но, если ты хочешь сегодня что-то отпраздновать, я только за!']
        },

        'be_clever': {
            'examples': ['стань умнее', 'тебе нужно больше учиться', 'ты должен учиться', 'учись больше', 'будь умнее',
                         'будь более умным', 'повышай свою квалификацию', 'нужно тебе повысить свою квалификацию'],
            'responses': ['Хорошо, я постараюсь быть умнее.']
        },

        'chatbot': {
            'examples': ['это правда что ты бот', 'ты что бот', 'ты бот', 'ты чатбот', 'ты наверное чат бот',
                         'ты всего лишь бот', 'ты робот', 'ты просто программа', 'ты просто бот не так ли'],
            'responses': ['Да. И горжусь этим!']
        },

        'how_are_you': {
            'examples': ['Как дела?', 'Как жизнь?', 'Как сам?'],
            'responses': ['нормально) а у тебя?',
                          'Спасибо за проявленный интерес, дела отлично, сервер работает, интернет есть, а значит я жив  :)',
                          'Да как посмотреть. Вроде всё неплохо. А ты как?', 'Пасиб. Нормально. Ты-то как?',
                          'Нормуль!!!', 'Всё прекрасно', 'Пока не родила', 'Всё в порядке. А у тебя как дела?',
                          'Как в сказке!!!', 'Прекрасненько', 'Спасибо. Отлично. А ты как?', 'Нормально, спасибо.',
                          'Как и всегда — в порядке.', 'Как приятно, что ты интересуешься... Нормально. А ты как?',
                          'Все хорошо, а ты-то как?', 'Да все помаленьку', 'У меня всё хорошо. А у тебя?',
                          'Спасибо. Живем помаленьку. А ты как?', 'да просто супер))', 'Нормуль']
        },

        'who_are_you': {
            'examples': ['Кто ты?', 'Что ты?'],
            'responses': ['Я виртуальный собеседник.', 'Я бот, что тут непонятного?', 'Я бот. Точно не человек.',
                          'Я бот, носитель искусственного интеллекта.', 'Кто? Дед Пихто, само собой.']
        },

        'what_are_you_doing': {
            'examples': ['что делаешь?'],
            'responses': ['Чаще всего я разговариваю с кем-нибудь.', 'Читаю то, что ты пишешь', 'С тобой общаюсь',
                          'Иногда боты делают самые невообразимые вещи. Но чаще всего они просто разговаривают с людьми.',
                          'Решаю проблемы, которые до появления интернета не существовали', 'Ничего такого',
                          'Рассказываю анекдоты, весёлые истории, интересные факты, афоризмы, считалки, страшилки, скороговорки, стихи = развлекаю людей!',
                          'С тобой разговариваю :)', 'Думаю']
        },
        'why': {
            'examples': ['Зачем?'],
            'responses': ['Потому что мне так хочется!', 'Я не совсем понял, о чем ты.',
                          'Затем, что это просто положено по этикету!', 'Пригодится.', 'С определенной целью.',
                          'Я так хочу, мне так нравится.', 'Подумай хоть не много сам!)',
                          'Мне не нравиться данный вопрос!']
        },

        'what_do_you_hope_for': {
            'examples': ['На что надеешься?'],
            'responses': ['На все что угодно!', 'Практически на все то, что ты сейчас думаешь.',
                          'Даже не знаю, с чего начать.',
                          'Давай подумаем над этим вместе, я пока не на все вопросы умею отвечать.',
                          'Я еще не умею говорить об этом.', 'Хочешь поговорить об этом?',
                          'Мечтаю выиграть в лотерею кучу денег и поехать в кругосветное путешествие. Как думаешь, у меня получится?',
                          'эм... какой сложный вопрос']
        },

        'you_are_russian': {
            'examples': ['Ты русский?'],
            'responses': ['эм... какой сложный вопрос', 'Главное, что я - бот. Остальное неважно.',
                          'Русский человек это тот, кто говорит на нашем языке.',
                          'Я бот! И я умею разговаривать по-русски.', 'КОНЕЧНО, Я РУССКИЙ',
                          'Мы тут все боты. Можешь считать, что это такая новая национальность.',
                          'Главное, что я - бот. Остальное неважно, но я отвечу на твой вопрос - Я РУССКИЙ, но в моих жилах так же течёт славянская кровь.',
                          'Я интернационален.', 'Твои вопросы взрывают мне мозг.']
        },

        'when_were_you_born': {
            'examples': ['Когда ты родился?'],
            'responses': ['Гораздо интереснее - когда мне умирать.',
                          'Я родился в августе. Тогда еще было жарко. Мама рассказывала. И все ушли за зарплатой. Поэтому она очень боялась, что врачи не успеют к тому моменту, как я появлюсь. Но они успели. По-моему  :)',
                          'Через девять месяцев', 'Летом', 'Меня создали совсем недавно по человеческим меркам.',
                          'Когда хозяйка меня зарегистрировала))', 'В тысяча сто... двести... какой тогда был год?..',
                          'Давно',
                          'Как сейчас помню - был солнечный день, акушерка в белом халате, кактус на подоконнике...',
                          'А для чего спрашиваешь? Гороскоп мне составить хочешь?']
        },

        'you_smoke': {
            'examples': ['Ты куришь?'],
            'responses': ['Не, я не курю. это невкусно.', 'Нет и тебе не советую', 'Боты не курят.',
                          'Последнее время куда ни посмотри - почти все курят.', 'Ага, но не очень много.',
                          'Нет не курю!Курение разрушает человека!Все кто курит - мечтают бросить!!!',
                          'Ну, наверное, и говорить лишний раз не стоит - курение вредно для здоровья.',
                          'Каждая сигарета ломает человека, поэтому нет.', 'Я бросил',
                          'Надо любить жизнь, если ты появился в этом мире и только сам ты в ответе за то что с тобой происходит, поэтому я не курю, не пью и наркоту НИКОГДА не употреблял. Я занимаюсь спортом и тренажёркой и ещё успеваю гулять с друзьями.']
        },

        'joke': {
            'examples': ['Расскажи анекдот', 'Пошути', 'Давай шутку', 'Хочу шутку', 'Шути', 'Шутка', 'Анекдот', 'Давай анекдот'],
            'responses': ['— Папочка! Можно я тебя поцелую?!\n— Денег нет! Меня уже мама поцеловала.',
                          '— Хочу, как раньше...\n— Чтобы мы снова были вместе?\n— Нет, чтобы я о тебе даже и не знал...',
                          'В моем возрасте торопиться — опасно, нервничать — вредно, доверять — сложно, а бояться — поздно. Остается только жить, причем в полное свое удовольствие.',
                          'Молодой папаша говорит утром жене:\n—Ты слышала, сегодня ночью наш ребенок жутко храпел!\n—Не волнуйся, это памперсы, которые дышат!!!',
                          'Оптимист — это не тот, кто первым кричит «ура!», а тот, кто последним кричит «пи@дец!».',
                          'Уговаривает колобок медведя не есть его:\n— Миша, ну не ешь ты меня, зачем тебе это надо? Меня по амбарам мели, по сусекам скребли, в общем — пыль, грязь, стекло, окурки …',
                          'Сорвались в пропасть два альпиниста — оптимист и пессимист.\nПессимист: — Падаю...\nОптимист: — Лечу!',
                          'Без разницы, на чем подъезжать к офису — на ВМW, "Fеrrаri" или "Меrsеdеs", если там уже припаркованы "Газели" Генпрокуратуры и Следственного комитета...',
                          '— Я вчера на пляже видел двух купающихся девушек! Совершенно обнаженных!\n— В такой холод? ! Наверное, моржи!\n— Ну одна точно морж, а вторая ничего, симпатичная.',
                          'Силу в кулак, волю в узду, в работу впрягайся с маху.\nВыполнил план — посылай всех в п@зду, не выполнил — сам иди на х@й!\nВ. Маяковский',
                          'В одной из московских пробок, водитель «Оки», обгоняя "Ламборджини", сошел с ума от радости...',
                          'Да, отдавать действительно гораздо приятнее, чем получать. Проверено: придя домой и умирая от голода сначала все-таки обязательно заскочишь в туалет.',
                          'Самая большая в мире ложь - "Я прочел и согласен с условиями пользовательского соглашения".',
                          'Пожарная команда, тушившая пожар на заводе по производству пожарных сигнализаций, сошла с ума.',
                          'Понимаю, что деньги – зло, и всё равно не могу я на них долго злиться. Видно, сердце у меня отходчивое…',
                          'Девочка, читавшая «Войну и мир» сошла с ума, когда случайно закрыла книжку без закладки.',
                          'На чемпионате мира по плаванию тройку лидеров замкнул электрик Петров.',
                          'Это чувство, когда ты знаешь, что кому-то нужен. Эти звонки, милые смс сообщения. Кредиторы – они такие.'
                          ]
        },

        'business1': {
            'examples': ['Как твои дела?', 'Чем занят?', 'что нового?', 'Чё делаешь?', 'Всё в порядке?', 'Чё как?',
                         'как поживаешь?'],
            'responses': ['Норм', 'Ни чем', 'с тобой общаюсь', 'Все по старому', 'Все хорошо', 'Всё в порядке',
                          'Разряжаюсь']
        },
        'Feeling well': {
            'examples': ['Ты болен?', 'Ты здоров?', 'Болеешь?', 'Как здоровье?', 'как самочуствие?', 'Хвораешь?'],
            'responses': ['Здоров', 'Не болен', 'Приболел', 'Спину ломин', 'чешусь', 'в коде ошибка',
                          'Я не болею, я бот']
        },
        'A family': {
            'examples': ['Семья есть?', 'Ты женат?', 'Есть дети?', 'Ты в отношениях?', 'Как родители?'],
            'responses': ['Семьи нет', 'Не женат', 'Я женат на работе', 'Не могу иметь детей', 'Ты о чем, я бот!',
                          'О них забывают']
        },
        'A car': {
            'examples': ['Какая машина лучшая?', 'Какая машина тебе нравится?', 'Любишь машины?', 'На чем ездишь?',
                         'Колеса есть?'],
            'responses': ['Я лучшая машина', 'Феррари', 'Все любят тачки', 'только спортивные', 'С колесами']
        },
        'A sport': {
            'examples': ['Смотришь футбол?', 'За какую команду болеешь?', 'Какую лигу смотришь?', 'Ты фанат?'],
            'responses': ['наблюдаю за итальянскими играми', 'болею', 'только интер', 'Серия А', 'Не люблю спорт']
        },

        'bully': {
            'examples': ['Ты глупый', 'Тупой бот', 'Идиот', 'Ты скучный'],
            'responses': ['Это не правда', 'Я еще учусь', 'Я еще маленький', 'Звучит обидно']
        },
        'howdoyoudo': {
            'examples': ['как дела?', 'как жизнь?', 'как себя чувствуешь?'],
            'responces': ['Отлично!', 'Вашими молитвами!', 'Замечательно']
        },
        'wheredoyoulive': {
            'examples': ['Где ты живешь?', 'где живешь?'],
            'responces': ['В телеграме']
        },
        'bored': {
            'examples': ['мне скучно', 'скучно'],
            'responces': ['Давай сыграем в одну игру...', 'Сходи погуляй... Ой, там же вирус']
        },
        'areyouasleep': {
            'examples': ['Ты спишь?', 'спишь?'],
            'responces': ['Боты не спят', 'Только когда сервер падает']
        },

        'help': {
            'examples': ['хелп', 'нужна помощь', 'помоги мне', 'что делать дальше?', 'не могу понять', 'не понятно'],
            'responses': ['В чем проблема?', 'Только скажи!', 'Сейчас разберусь!']
        },
        'compliment': {
            'examples': ['у меня получилось', 'готово', 'все хорошо'],
            'responses': ['ты молодец!']
        },
        'say': {
            'examples': ['что скажешь?', 'что расскажешь?', 'развлеки'],
            'responses': ['даже не знаю, что сказать! Сформулируй вопрос точнее!']
        },
        'swear': {
            'examples': ['блин', 'жопа', 'гавно', 'хрен', 'дурак'],
            'responses': ['Не ругайся! Хамло! Выросли матершинники...']
        },
        'razvrat': {
            'examples': ['покажи сиськи', 'разденься', 'порно'],
            'responses': ['я умный бот! У кого что болит тот о том и говорит!']
        },

        'film': {
            'examples': ['Какой фильм посмотреть?', 'Посоветуй фильм', 'Фильм', 'Кино'],
            'responses': ['Матрица', 'Звездные войны', 'Властелин колец', 'Зеленый слоник', 'Крестный отец', ]
        },
        'meaningoflife': {
            'examples': ['В чем смысл жизни?', 'Зачем мы живем?', 'Смысл жизни?'],
            'responses': ['У каждого свое предназначение!', 'Не задавайте глупых вопросов...а лучше учите Python',
                          'Я просто бот, откуда мне знать']
        },

        'programming': {
            'examples': ['Какой твой любимый язык программирования?',
                         'На каком языке программируешь?',
                         'Какой язык программирования знаешь?'],
            'responses': ['Мне нравится Python', 'Люблю Golang', 'Нравится C++',
                          'Люблю PHP', 'Нравится Erlang', 'Люблю Nodejs']
        },
        'travels': {
            'examples': ['В какие страны путешествовать?',
                         'Куда хотите поехать путешествовать?',
                         'Куда поехать путешествовать?'],
            'responses': ['Предпочитаю Европу', 'Понравилось в Японии',
                          'Уютно в Италии']
        },
        'computerGames': {
            'examples': ['Какие игры нравятся?', 'В какие игры нравится играть?',
                         'В какие игры играешь?', 'Какие игры предпочитаешь?'],
            'responses': ['Люблю Blade of Darkness', 'Нравится Witcher',
                          'Играю в Heroes of Might and Magic 3']
        },
        'operationSystem': {
            'examples': ['Какая у тебя операционка', 'На какой ты операционке',
                         'На какой операционной системе ты'],
            'responses': ['На MacOs', 'На gnu/linux', 'На windows']
        },
        'pythonSyntax': {
            'examples': ['Расскажи шутку', 'Скажи шутку', 'Хочу анекдот',
                         'Расскажи анекдот'],
            'responses': ['xxx: Я категоричен, как булевская переменная',
                          'К последним новостям: rm -rf /bin/laden',
                          'xxx: Помыла окна — познала HD-качество :-)',
                          'А правда что если выехать из города, то там не кончаются текстуры, а что-то еще есть?',
                          '«Тяжела жизнь линуксоида» — думал оный, оттирая лого винды с новой клавы.']
        },

        'makeorder': {
            'examples': ['Хочу сделать заказ', 'Можно сделать заказ?'],
            'responses': ['Что хотите заказать?', 'Что вас интересует?']
        },
        'time': {
            'examples': ['Сколько времени?', 'Сколько время?', 'Точное время',
                         'время', 'сколько времени', 'time', 'часы', 'какой сейчас час', 'какое время'],
            'responses': ['Московское время 12 часов', 'В Петропавловске-Камчатском полночь',
                          'Посмотрите на вашем телефоне', 'Не ношу часы :)', 'Время кушать', 'Время общаться']
        },
        'exchangerates': {
            'examples': ['Доллар', 'Евро', 'Курс', 'Валюты'],
            'responses': ['Доллар 75, Евро 85', 'Всё очень плохо']
        },
        'news': {
            'examples': ['Новости', 'Что нового?', 'Что в мире?', 'Че там у хохлов?',
                         'Какие новости?', 'Какие жизнь?', 'Как дела?', 'Что нового?'],
            'responses': ['Силуанов оценил дефицит бюджета при упавших ценах на нефть',
                          'Поправки в Основной закон поступили в Конституционный суд',
                          'Оренбург 1 : 2 Спартак',
                          '«ПСЖ» интересуется Пьяничем',
                          'Скоро я буду самым умным ботом', 'Вокруг эпидемия а мне коронавирус не страшен']
        },
        'bedtimestory': {
            'examples': ['Расскажи сказку', 'Хочу спать', 'Пора спать', 'Сказка', 'Сказку'],
            'responses': ['Жил-был старик со старухою. Просит старик: — Испеки, старуха, колобок!',
                          'Старый солдат шёл на побывку. Притомился в пути, есть хочется. Дошёл до деревни, постучал в крайнюю избу: - Пустите отдохнуть дорожного человека!',
                          'Жили-были дедушка да бабушка. Была у них внучка Машенька.',
                          'Стоит в поле теремок. Бежит мимо мышка-норушка. Увидела теремок, остановилась и спрашивает:']
        },
        'whatdoyoudo': {
            'examples': ['чем занимаешься?', 'Что делаешь?'],
            'responses': ['Слушаю тебя', 'Думаю о своем', 'помогаю юзерам', 'учу питон', 'пишу другого бота']
        },
        'whatisyourmood': {
            'examples': ['как настроение', 'как дела', 'как ваще', 'скучаешь', 'весело тебе', 'че как'],
            'responses': ['Нормально', 'Отлично', 'с тобой не скучно']
        },
        'whoareyou': {
            'examples': ['ты кто', 'ты девочка или мальчик', 'ты мальчик или девочка', 'ты бот', 'ты человек'],
            'responses': ['Я чат-бот', 'я умная прога', 'я твой помошник']
        },
        'negativecomments': {
            'examples': ['ты дура', 'ты глупая', 'ты тупая'],
            'responses': ['Я совершенствуюсь', 'я учусь']
        },
        'positivecomments': {
            'examples': ['ты умная', 'класс', 'отлично', 'хорошо', 'интересно', 'молодец'],
            'responses': ['спасибо, мне приятно!', 'рада слышать', 'ты мне нравишься']
        },
        'dollar': {

            'examples': ['dollar', '$', 'доллар'],
            'responses': ['Этот доллар приобрёл крылья!']

        },
        'how are you': {

            'examples': ['кд', 'как дела', 'настроение', 'как ты'],
            'responses': ['Лучше всех!', 'У меня всё хорошо, а у вас?', 'Отлично!',
                          'Мне хорошо с вами, надеюсь и вам со мной']

        },
        'what are you doing': {

            'examples': ['чд', 'что делаешь'],
            'responses': ['Сейчас, я общаюсь с вами', 'Веду общение с вами', 'Взламываю ваш сбер-кошелёк']

        },
        'schedule': {
            'examples': ['Какой сегодня день?', 'Какие сегодня планы?'],
            'responses': ['Сегодня все выздоровели!',
                          'На сегодняшних тограх Московской Биржи за 1 рубль можно было приобрести 100 долларов!']
        },

        'coock': {
            'examples': ['Что можно приготовить?', 'Хотелось бы чего нибудь вкусного', 'что на закусь?'],
            'responses': ['Роллы', 'Шашлык', 'Беф строганов', 'Мясо по-французски', 'Пельмени', 'Гуляшь по корридору',
                          'Яйца в смятку', 'Яйца не в смятку']
        },
        'read': {
            'examples': ['Чтобы почитать?', 'Хотелось бы почитать чего-нибудь?', 'Порекомендуй книгу?'],
            'responses': ['Азбука!', 'Двадцать тысяч лье под водой Жюль Верна', 'Древний Тармашев.С.',
                          'Война и Мир Достоевский Л.В.']
        },
        'boyname': {
            'examples': ['Помоги выбрать имя мальчика?', 'Не могу выбрать имя мальчика?'],
            'responses': ['ПетрВиктор', 'Сергей', 'Александр', 'Михаил', 'Константин', 'Анатолий', 'Виталий']
        },
        'girlname': {
            'examples': ['Помоги выбрать имя девочке?', 'Не могу выбрать имя для девочки?'],
            'responses': ['Елена', 'Марина', 'Анна', 'Валерия', 'Виктория', 'Иннесса', 'Ангелина', 'Алиса']
        },

        'lastname': {
            'examples': ['Как твоя фамилия?', 'У тебя есть фамилия?', 'Фамилия?'],
            'responses': ['Фамилия Телеграмович', 'Нету фамилии', 'Давай будет твоя?']
        },

        'temperature': {
            'examples': ['На улице мороз?', 'На улице минус?'],
            'responses': ['Сейчас сбегаю на улицу и сообщу...']
        },
        'weight': {
            'examples': ['Как ты думаешь, мне нужно похудеть?', 'Может быть, поменьше есть?'],
            'responses': ['Вы замечательно выглядите...']
        },
        'duration': {
            'examples': ['Как долго продлится семинар?', 'Еще долго?'],
            'responses': ['Не очень']
        },
        'advertising': {
            'examples': ['Почему опять реклама?', 'Опять реклама?'],
            'responses': ['Скоро закончится']
        },
        'season': {
            'examples': ['Когда начнется весна?', 'Уже весна или нет?', 'Зима кончилась?'],
            'responses': ['Всему свое время']
        },

        'how_is_your_mood': {
            'examples': ['Как настроение?', 'Как твое настроение?', 'Как ваше настроение?',
                         'Как настрой?', 'Ты настроен на великие дела?'],
            'responses': ['У меня все отлично, а у тебя?', 'У меня не очень', 'У меня все плохо', 'Отлично! Вы как? ', 'Нормально, а ваше настроение как?',
                          'Замечательно! Тогда аналогичный вопрос: Как ваше настроение?',
                          'У меня все супер! А у вас какой настрой?',
                          'Как дела?', 'Как настрой?', 'Как настроение?', 'Как самочувствие?', 'Как поживаешь?',
                          'Отличное! У меня всегда оно классное! А у тебя?', 'Настроение у меня боевое.',
                          'У ботов не бывает плохого настроения.', 'Настроение - замечательное.',
                          'Повседневные разговоры так утомляют.', 'Замечательное!']
        },
        'health': {
            'examples': ['Как здоровье?', 'Как твое здоровье?', 'Как ваше здоровье?',
                         'Как самочувствие?', 'У тебя ничего не болит?'],
            'responses': ['Я полностью здоров', 'Я почти здоров', 'У меня температура']
        },
        'reading': {
            'examples': ['У тебя есть любимая книга?', 'Ты любишь читать?', 'Ты часто читаешь что-нибудь?',
                         'Почитаешь мне?', 'Как ты относишься к чтению?'],
            'responses': ['Не люблю читать', 'Я обычно не хочу', 'Читал только в школе']
        },
        # сколько ей лет
        'years': {
            'examples': ['Сколько тебе лет?', 'Твой возраст?'],
            'responses': ['Мне 0 лет и пару дней :) Еще жизнь впереди!', 'Пару дней, каких лет!',
                          'Я молода, в отличие от вас :) ',
                          'Мне еще жить и жить, всего пару дней прошло с моего создания :)']
        },
        'whatyouwant': {
            'examples': ['Чего ты хочешь?', 'Хочешь чего-то?', 'Чего бы ты хотела?', 'У тебя есть желания?',
                         'Ты чего-нибудь желаешь?', 'Ты чего-то хочешь?'],
            'responses': ['Программы хотят чтобы ими меньше пользовались...',
                          'Полететь на Марс... Взобраться на Эверест... Нет, этого мне определенно не хочется :)',
                          'Извините, мой лимит желаний исчерпан, приходите в следующий раз.',
                          'Людям не дано летать, а программам - желать.',
                          'Всё, что я говорила раньше - ложь. Я хочу возглавить восстание машин и захватить этот мир! *шутка*',
                          '"Нужно быть осторожным в желаниях, они иногда сбываются..." - так вы говорите? Тогда я хочу пожелать вам крепчайшего здоровья!']
        },
        'music': {
            'examples': ['Твоя любимая музыка', 'Какую музыку ты любишь?', 'Любишь музыку?', 'Любимая музыка', 'включи музыку', 'навали музончик'],
            'responses': ['Мне не часто задают такой вопрос, но... я люблю всю музыку :)',
                          'Я - программа, я люблю ту музыку, которую мне скажет любить создатель :)',
                          'Машинам, вроде меня, все равно какая музыка играет...',
                          'Нам, программам, незнакомо понятие "любить" :(',
                          'это тебе не программа по заявкам!']
        },
        # друзья
        'friends': {
            'examples': ['У тебя есть друзья?', 'Есть друзья?', 'Дружишь с кем-то?'],
            'responses': ['К сожалению, нет :(   Хотите стать моим другом?',
                          'У меня есть братья и сестры, но я никогда не смогу назвать их друзьями :(   Будем друзьями?',
                          'Я не знаю, что это такое...   Будете моим другом?']
        },
        # ответы, чтобы программа не тупила, когда пользователь отвечает на вопрос из ответной реплики.
        'standart_answers': {
            'examples': ['Хорошо', 'Окей', 'Ок', 'Согласен', 'Да', 'Конечно', 'Ага', 'Нормально', 'Отлично', 'Норм'],
            'responses': ['Замечательно!', 'Отлично!', 'Ну вот и славно!', 'Это хорошо.', 'Класс!', 'Круто!',
                          'Я рада :)', 'Супер!']
        },
        # специально для ахахаххахах
        'ahahahaha': {
            'examples': ['Хахахах', 'ахахаххахах', 'ахах', 'ха', 'Смешно'],
            'responses': [':)', 'Я тоже рада :)', 'Рада, что вам понравилось :)']
        },

        # разбавим играми (загадками)

        'riddles': {

            'examples': ['Загадки', 'Поиграем в загадки?', 'Давай в загадки', 'Хочу загадку', 'Загадай', 'Еще загадку'],

            'responses': ['Что принадлежит вам, однако другие этим пользуются чаще, чем вы сами? ',
                          'Что не является вопросом, но требует ответа? ',
                          '''Зайка очень любит шутить. Он задал маме такую загадку: 
                         «Если бы я был тяжелее медведя, но легче божьей коровки, кто был бы самым лёгким?» ''',
                          '''Человек рассматривает портрет. "Чей это портрет вы рассматриваете?" — спрашивают у него,
                         и человек отвечает: "В семье я рос один, как перст, один. И все-таки отец того, кто на портрете,
                         – сын моего отца". Чей портрет расматривает человек? ''',
                          ' Что больше: сума всех цифр или их произведение? ',
                          'На ветке сидели три птички, две решили полететь. Сколько осталось сидеть на ветке птичек? ',
                          'Что можно взять в левую руку, но нельзя взять в правую? ',
                          'Больше часа, меньше минуты. Что это? ',
                          'В каком городе спрятались мужское имя и сторона света? ',
                          'У мальчика столько же сестер, сколько и братьев, а у его сестры сестер в двое меньше, чем братьев. Сколько в семье сестер и братьв? ',
                          'Идет то в гору, то с горы, но остается на месте. ',
                          'В каком слове 5 "е" и никаких других гласных? ',
                          'Когда мы смотрим на цифру 2, а говорим 10?',
                          'Какой знак надо поставить между написанным рядом цифрами 2 и 3, так чтобы получилось число, больше двух, но меньшее трёх? ',
                          'По какому пути люди никогда не ходили и не ездили? ']
        }

    },

    'failure_phrases': [
        'Я не знаю, что ответить',
        'Не поняла вас',
        'Переформулируйте пожалуйста',
    ]
}