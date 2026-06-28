import "@testing-library/jest-dom"

// jsdom doesn't implement IntersectionObserver, which framer-motion's
// whileInView relies on. Provide a no-op mock so components render in tests.
class IntersectionObserver {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return []
  }
}
globalThis.IntersectionObserver = IntersectionObserver
globalThis.IntersectionObserverEntry = function () {}
