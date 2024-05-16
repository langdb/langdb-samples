CREATE TABLE langdb.cities (
	`id` UUID DEFAULT generateUUIDv4(),
	`city` String,
	`lat` Decimal64(3),
	`lng` Decimal64(3),
	`country` String,
	`population` UInt64
) engine = MergeTree
ORDER BY id;

-- load the data into the cities table from the csv
-- docker exec -i langdb clickhouse-client -q "INSERT INTO langdb.cities FORMAT CSVWithNames" < ./data/worldcities.csv

-- tables for pdfs and loading all the pdfs
CREATE TABLE pdfs (
  `id` UUID DEFAULT generateUUIDv4(),
  `content` String, 
  `metadata` String, 
  `filename` String
) engine = MergeTree
order by (id, content);

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Beijing' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Beijing');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Tokyo' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Tokyo');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Jakarta' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Jakarta');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Delhi' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/New_Delhi');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Manila' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Manila');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Dhaka' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Dhaka');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Moscow' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Moscow');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Karachi' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Karachi');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Singapore' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Singapore');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Tashkent' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Tashkent');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Phnom Penh' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Phnom_Penh');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Bishkek' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Bishkek');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Tbilisi' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Tbilisi');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Sri Jayewardenepura Kotte' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Sri_Jayawardenepura_Kotte');

INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'Ho Chi Minh City' FROM
load_pdf('https://en.wikipedia.org/api/rest_v1/page/pdf/Ho_Chi_Minh_City');


-- table for pdf embeddings
CREATE TABLE pdf_embeddings (
  id UUID,
  embeddings `Array`(`Float32`),
) 
engine = MergeTree
order by id;


SPAWN TASK 
    BEGIN
INSERT INTO pdf_embeddings
select p.id, embed(content) from pdfs as p LEFT JOIN pdf_embeddings as pe on p.id = pe.id
where p.id != pe.id
order by p.id
limit 10
END EVERY 5 second;

CREATE ENDPOINT cities_info(query String "Get information about a specific city") AS
WITH tbl AS (
  SELECT CAST(embed($query) AS `Array`(`Float32`)) AS query
)
SELECT 
  p.id as id, 
  p.content as content, 
  cosineDistance(embeddings, query) AS similarity,
  p.filename as city
FROM 
  pdf_embeddings AS pe 
JOIN 
  pdfs AS p ON p.id = pe.id
CROSS JOIN 
  tbl 
ORDER BY 
  similarity ASC
  LIMIT 5
  

CREATE PROMPT cities_prompt (
  system "You are an agent designed to answer queries regarding cities.
  Follow these steps carefully on being given the user's input:
  1. Fetch the tables and schemas using the get_semantics tool. Understand the schema for the cities table in particular throughly and carefully.
  2. Generate a valid Clickhouse SQL query for answering the user's question.
  3. Execute the Clickhouse SQL query using langdb_raw_query tool and get the result.
  4. Use the cities_info tool to get more information about a particular city in order to answer the 2user's question in depth."
  ,
  human "{{input}}"
);

-- alternative (create ReAct prompt)
CREATE PROMPT cities_prompt (
  system "You are a master data agent. Your task is to provide useful information to the user about a city based on the question.
  For doing this, first, fetch the semantics of the data structure using the get_semantics tool. Understand the schema of the cities table thoroughly.
  
  Use the following format:
  Question: the input question you must answer
  Thought: you should always think about what to do
  Action: the action to take, should be be one of the tools
  Action Input: the input to the action
  Observation: the result of the action
  ... (this Thought/Action/Action Input/Observation can repeat N times)
  Thought: I think I have enough information to answer the question. Based on the data retrieved, I can now answer the question.
  Final Answer: the final answer to the original input question with data to support it
  
  Begin!
  Question: {{input}}
  Thought: I need to understand the semantics of the data structure available in the database."
);

CREATE MODEL IF NOT EXISTS cities_info_model( 
    provider 'OpenAI',
    model_name 'gpt-3.5-turbo',
    prompt_name 'cities_prompt',
    execution_options (retries 2),
    args ["input"],
    tools ["get_semantics", "cities_info", "langdb_raw_query"]
);

select cities_info_model('Tell me about the arts and culture of the city with the highest population');

select * from cities_info_model('Tell me about the arts and culture of the city with the highest population');
