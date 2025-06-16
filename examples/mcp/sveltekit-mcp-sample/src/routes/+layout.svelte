<script>
  import { afterNavigate } from '$app/navigation';
  import SiteHeader from '$lib/components/SiteHeader.svelte';

  afterNavigate(({ from, to, type }) => {
    console.log(`navigated from ${from?.url.pathname} to ${to.url.pathname}`);

    // scroll restoration
    window.scrollTo(0, 0);

    // simple analytics ping
    fetch('/api/analytics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        from: from?.url.pathname,
        to: to.url.pathname,
        ts: Date.now(),
      })
    });
  });
</script>

<SiteHeader />

<main class="p-8">
  <slot />
</main>
