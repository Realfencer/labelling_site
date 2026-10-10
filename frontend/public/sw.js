// The Chutes low-balance notification feature was removed. This stub only unregisters the old
// service worker that earlier versions installed in browsers.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(self.registration.unregister());
});
