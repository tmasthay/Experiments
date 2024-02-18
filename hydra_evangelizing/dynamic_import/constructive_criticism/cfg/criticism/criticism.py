def critique_focus(intensity=0):
    print(
        "You seem to be losing focus on the big picture. It is one tiny unit"
        " test on a function that rarely gets called. Move on to something"
        " more important."
    )
    if intensity >= 1:
        print('I mean for the love of God! You are wasting time!')
    if intensity >= 2:
        print('You are the worst!')
    if intensity >= 3:
        print('I wish you would just quit your freaking job already!')


def critique_commit_messages(intensity=0):
    print(
        "Your commit messages are very vague. They should be more descriptive"
        " and informative."
    )
    if intensity >= 1:
        print(
            'I mean for the love of God! Nobody on the team knows what you are'
            ' doing!'
        )
    if intensity >= 2:
        print(
            'You are the worst! I feel like Ryan Gosling in the Papyrus SNL'
            ' skit reading your commit messages!'
        )
    if intensity >= 3:
        print(
            'You are just as unemployable as the graphic designer for Avatar!'
        )
