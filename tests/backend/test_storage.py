"""TDD for app.db — User/Query SQLite store."""


def test_create_and_get_user(repo):
    user = repo.create_user("a@b.com", "hashed", points=14)
    assert user.points == 14
    assert user.uid
    assert repo.get_user_by_email("a@b.com").uid == user.uid
    assert repo.get_user_by_uid(user.uid).email == "a@b.com"


def test_create_user_duplicate_email_returns_none(repo):
    repo.create_user("a@b.com", "h", points=14)
    assert repo.create_user("a@b.com", "h2", points=14) is None


def test_decrement_points(repo):
    user = repo.create_user("a@b.com", "h", points=14)
    assert repo.decrement_points(user.uid) == 13
    assert repo.get_user_by_uid(user.uid).points == 13


def test_set_points(repo):
    user = repo.create_user("a@b.com", "h", points=14)
    repo.set_points(user.uid, 3)
    assert repo.get_user_by_uid(user.uid).points == 3


def test_create_query_and_rating(repo):
    user = repo.create_user("a@b.com", "h", points=14)
    qid = repo.create_query(
        uid=user.uid,
        question="is aspirin safe?",
        opinion="3",
        medical="True",
        news="False",
        label="Trustworthy",
        response="Yes, generally.",
    )
    q = repo.get_query(qid)
    assert q.label == "Trustworthy"
    assert q.rating == "0"
    assert repo.set_rating(qid, "5") is True
    assert repo.get_query(qid).rating == "5"


def test_get_query_missing_returns_none(repo):
    assert repo.get_query("does-not-exist") is None


def test_set_rating_missing_returns_false(repo):
    assert repo.set_rating("nope", "5") is False
