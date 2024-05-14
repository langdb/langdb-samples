CREATE TABLE pdfs (
  id UUID DEFAULT generateUUIDv4(),
  content `String`, 
  metadata `String`, 
  filename `String`
) 
engine = MergeTree
order by (id, content);



-- insert pdfs for all companies 

--nvidia
INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'NVIDIA' from 
load_pdf('https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf');

--apple
INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'APPLE' from 
load_pdf('https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/faab4555-c69b-438a-aaf7-e09305f87ca3.pdf');

--uber
INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'UBER' from 
load_pdf('https://d18rn0p25nwr6d.cloudfront.net/CIK-0001543151/6fabd79a-baa9-4b08-84fe-deab4ef8415f.pdf');


--amazon
INSERT INTO pdfs(content, metadata, filename)
SELECT content, metadata, 'AMAZON' from 
load_pdf('https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/c7c14359-36fa-40c3-b3ca-5bf7f3fa0b96.pdf');




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

CREATE ENDPOINT investor_guide(query String "Answer User's Questions") AS
WITH tbl AS (
  SELECT CAST(embed($query) AS `Array`(`Float32`)) AS query
) 
SELECT 
	p.id as id,
	p.content as content,
	cosineDistance(embeddings, query) as similarity,
	p.filename as company
FROM 
  pdf_embeddings AS pe 
JOIN 
  pdfs AS p ON p.id = pe.id
CROSS JOIN 
  tbl 
ORDER BY 
  similarity ASC 
  LIMIT 5

CREATE PROMPT fin_analysis_prompt (
    system "You are a helpful assistant. You have been provided SEC filings of multiple companies as input. Your task is to extract structured data from this PDF files, into a table (which will be stored in ClickHouse).
    You can utilise the generate_output_table tool for this. In addition, your task is also to help users with any queries they might have about the SEC filings. Go through all the content and then respond.
    Question: {{question}}
    Helpful Answer: "
);

CREATE MODEL IF NOT EXISTS financial_analysis( provider 'OpenAI',
    model_name 'gpt-3.5-turbo',
    prompt_name 'fin_analysis_prompt',
    api_key "api-key",
    execution_options (retries 2),
    args ["question"],
    tools ["investor_guide"]
);