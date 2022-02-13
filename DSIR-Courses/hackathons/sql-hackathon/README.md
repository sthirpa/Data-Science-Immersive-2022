# SQL Hackathon 🎉

## Goal

Perform data analysis on Kiva Loans. See what insights you can find as you develop your SQL muscles! 💪

This dataset is a random subset of rows and selected column from [this Kaggle dataset](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding#kiva_loans.csv). The original dataset is publicly available under a [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/).

## Roles

- One person from your group will be the spokesperson who will present your results.
- Another person is the moderator who will keep the group on task and make sure everything is humming along and everyone is on board with decisions and understands what's happening.
- The third person will share their screen and keep the official queries file for the group (everyone should work on their own queries, too).
- Fourth person? In charge of fun!

## Instructions

1. Make sure you have `pandasql` installed.

2. In the attached notebook, use queries to explore the data. You are doing analysis in this hackathon, you aren't predicting anything. See what interesting insights you can find. 😀

3. While putting the data into Python or Tableau would allow for cool visualizations, that's not our focus today. __See what insights you can learn using SQL only.__ There is a data dictionary [here](./data_dictionary.md).

4. Lightning presentations. After ~1.5 hours we'll reconvene and your group will do a 2-3 minute walk-through of your analysis. Assume the audience here is your classmates and instructors, to keep it simple.


## Have fun and good luck! 🚀


## Bonus

Woohoo! Got some fascinating insights to share and ready for some more fun?! If that sounds like you, and you feel confident and comfortable working with SQL queries in jupyter notebooks with pandasql, then step right up and take a minute to explore using PostgreSQL and pgAdmin4, available for download via [EDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads). Time to create a database on your locally hosted server to connect to and start querying in the browser...

1. In PgAdmin4, create the database, schema, and tables. In the left side bar, under the _Postgres_ server right click on _Databases_ and click _Create->Database_. Enter _kiva_ for the name and click the _Save_ button. Then expand the new database in the left side bar. Then bring up the query editor and enter [the code below](#code-to-create-schema-and-tables) in the query editor and run it.

2. Create 4 tables with the corresponding names to the csv files. After you have created the four tables, insert the data into the tables.

- Navigate to on one of the tables you created in the side bar. Right click on it.
- Select _import/export_
- __Toggle the _export_ button to _import___.
- Click the three dots and navigate to your cloned repo with the data files.
- Select the matching .csv file.
- __Toggle the _header_ button to *Yes*__.
- Click _Ok_.
- You should see a success message.

Repeat the import process for the other three tables.

3. Get [querying](https://www.pgadmin.org/docs/pgadmin4/development/query_tool.html)!


## Code to create schema and tables

```sql

-- Create database, schema, and tables

CREATE SCHEMA kiva
    AUTHORIZATION postgres;

CREATE TABLE kiva.financials
(
    "id" integer NOT NULL,
    "funded_amount" text,
    "currency" text,
    "term_in_months" float,
    PRIMARY KEY ("id")
)
WITH (
    OIDS = FALSE
);

ALTER TABLE kiva.financials
    OWNER to postgres;

CREATE TABLE kiva.use
(
    "id" integer NOT NULL,
    "activity" text,
    "sector" text,
    "use" text,
    PRIMARY KEY ("id")
)
WITH (
    OIDS = FALSE
);

ALTER TABLE kiva.use
    OWNER to postgres;


CREATE TABLE kiva.demographics
(
    "id" integer NOT NULL,
    "country" text,
    "region" text,
    "borrower_genders" text,

    PRIMARY KEY ("id")
)
WITH (
    OIDS = FALSE
);


ALTER TABLE kiva.demographics
    OWNER to postgres;


CREATE TABLE kiva.crowdsource
(
    "id" integer NOT NULL,
    "posted_time" text,
    "funded_time" text,
    "lender_count" integer,

    PRIMARY KEY ("id")
)
WITH (
    OIDS = FALSE
);

ALTER TABLE kiva.crowdsource
    OWNER to postgres;**
```
