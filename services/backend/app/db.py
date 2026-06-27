"""SQLite persistence for users, query history, and ratings.

Replaces the legacy Firebase Realtime DB. A Query row's id is the ``key`` /
``reference`` returned to the client and later used to attach a rating.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String, create_engine, desc, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    uid: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    points: Mapped[int] = mapped_column(Integer, default=0)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    uid: Mapped[str] = mapped_column(String, index=True)
    question: Mapped[str] = mapped_column(String)
    opinion: Mapped[str] = mapped_column(String, default="0")
    medical: Mapped[str] = mapped_column(String, default="True")
    news: Mapped[str] = mapped_column(String, default="False")
    label: Mapped[str] = mapped_column(String, default="Doubtful")
    response: Mapped[str] = mapped_column(String, default="")
    rating: Mapped[str] = mapped_column(String, default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


def init_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


class Repository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        # expire_on_commit=False keeps returned ORM objects usable after commit.
        self._Session = sessionmaker(engine, expire_on_commit=False, future=True)

    # ---- users ----
    def create_user(
        self,
        email: str,
        password_hash: str,
        points: int,
        uid: str | None = None,
        is_admin: bool = False,
    ) -> User | None:
        user = User(
            uid=uid or uuid.uuid4().hex,
            email=email,
            password_hash=password_hash,
            points=points,
            is_admin=is_admin,
        )
        with self._Session() as session:
            session.add(user)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return None
            return user

    def get_user_by_email(self, email: str) -> User | None:
        with self._Session() as session:
            return session.scalar(select(User).where(User.email == email))

    def get_user_by_uid(self, uid: str) -> User | None:
        with self._Session() as session:
            return session.get(User, uid)

    def list_users(self) -> list[User]:
        with self._Session() as session:
            return list(session.scalars(select(User).order_by(User.created_at)).all())

    def set_admin(self, uid: str, is_admin: bool) -> None:
        with self._Session() as session:
            user = session.get(User, uid)
            if user is not None:
                user.is_admin = is_admin
                session.commit()

    def set_points(self, uid: str, points: int) -> None:
        with self._Session() as session:
            user = session.get(User, uid)
            if user is not None:
                user.points = points
                session.commit()

    def decrement_points(self, uid: str) -> int:
        with self._Session() as session:
            user = session.get(User, uid)
            if user is None:
                return 0
            user.points = max(0, user.points - 1)
            session.commit()
            return user.points

    # ---- queries / ratings ----
    def create_query(
        self,
        *,
        uid: str,
        question: str,
        opinion: str,
        medical: str,
        news: str,
        label: str,
        response: str,
    ) -> str:
        query = Query(
            id=uuid.uuid4().hex,
            uid=uid,
            question=question,
            opinion=opinion,
            medical=medical,
            news=news,
            label=label,
            response=response,
        )
        with self._Session() as session:
            session.add(query)
            session.commit()
            return query.id

    def get_query(self, query_id: str) -> Query | None:
        with self._Session() as session:
            return session.get(Query, query_id)

    def set_rating(self, query_id: str, rating: str) -> bool:
        with self._Session() as session:
            query = session.get(Query, query_id)
            if query is None:
                return False
            query.rating = rating
            session.commit()
            return True

    def recent_queries(self, limit: int = 50) -> list[Query]:
        with self._Session() as session:
            stmt = select(Query).order_by(desc(Query.created_at)).limit(limit)
            return list(session.scalars(stmt).all())

    def count_users(self) -> int:
        with self._Session() as session:
            return session.scalar(select(func.count()).select_from(User)) or 0

    def count_queries(self) -> int:
        with self._Session() as session:
            return session.scalar(select(func.count()).select_from(Query)) or 0
