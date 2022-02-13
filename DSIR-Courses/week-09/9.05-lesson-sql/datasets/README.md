## Create tables


```sql
-- categories

CREATE TABLE public.categories (
    id bigint NOT NULL,
    name character varying(255)
);

ALTER TABLE public.categories OWNER TO postgres;

-- foods

CREATE TABLE public.foods (
    id bigint NOT NULL,
    name character varying(255),
    calories integer,
    carbs integer,
    fat integer,
    restaurant_id bigint
);

ALTER TABLE public.foods OWNER TO postgres;

--restaurants

CREATE TABLE public.restaurants (
    id bigint NOT NULL,
    name character varying(255)
);

ALTER TABLE public.restaurants OWNER TO postgres;

-- categories_foods

CREATE TABLE public.categories_foods (
    id bigint NOT NULL,
    food_id bigint,
    category_id bigint
);

ALTER TABLE public.categories_foods OWNER TO postgres;

-- Create read-only user
CREATE USER dsi_student WITH PASSWORD 'dsir0cks';

-- Grant access to current tables and views
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dsi_student;

-- Now make sure that's also available on new tables and views by default
ALTER DEFAULT PRIVILEGES
    IN SCHEMA public -- omit this line to make a default across all schemas
    GRANT SELECT
ON TABLES
TO dsi_student;

```
