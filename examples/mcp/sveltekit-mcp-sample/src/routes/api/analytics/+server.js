/** @type {import('@sveltejs/kit').RequestHandler} */
export async function POST({ request }) {
  const event = await request.json();
  console.log('Analytics event:', event);
  // You’d forward this to your real analytics back-end here
  return new Response(null, { status: 204 });
}
