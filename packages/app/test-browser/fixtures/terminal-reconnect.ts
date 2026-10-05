import { expect, mock, test } from "bun:test"
import { createRequire } from "node:module"
import { createComponent } from "solid-js"
import { render } from "solid-js/web"
import { OpenCode } from "@opencode/client/promise"
import { createKeyed } from "../../../gui-extensions/src/sdk/reactive"

// Use the same Solid transform already used by the model-selection fixture.
const require = createRequire(import.meta.url)

const solid = createRequire(require.resolve("vite-plugin-solid"))

const { transformSync } = solid("@babel/core")

Bun.plugin({
  name: "terminal-reconnect-solid",
  setup(build) {
    build.onLoad({ filter: /[\\/]terminal[\\/]terminal\.tsx$/ }, async (args) => ({
      contents: transformSync(await Bun.file(args.path).text(), {
        filename: args.path,
        presets: [solid.resolve("babel-preset-solid"), solid.resolve("@babel/preset-typescript")],
      }).code,
      loader: "js",
    }))
  },
})

mock.module("@opencode/ui/theme/context", () => ({
  useTheme: () => ({ mode: () => "dark", themes: () => ({}), themeId: () => "default" }),
}))

mock.module("../../../gui-extensions/src/sdk", () => ({
  createKeyed,
  useExtension: () => ({
    appearance: { font: () => "monospace" },
    keybinds: { matches: () => false },
    system: { openExternal() {} },
    t: (key: string) => key,
  }),
}))

mock.module("@opencode/ui/toast", () => ({ showToast() {} }))

const { Terminal } = await import("../../../gui-extensions/src/terminal/terminal")

class Socket extends EventTarget {
  binaryType: WebSocket["binaryType"] = "blob"
  readyState: number = WebSocket.OPEN
  close() {
    this.readyState = WebSocket.CLOSED
  }
  send() {}
}

test("transient failures retry the same PTY and preserve the native ticket-first handshake", async () => {
  const tickets: string[] = []
  const opened: string[] = []
  const cloned: Error[] = []
  const inspected: string[] = []
  const socket = new Socket()

  const api = OpenCode.make({
    baseUrl: "https://server.example",
    fetch: Object.assign(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const request = input instanceof Request ? input : new Request(input, init)
        tickets.push(request.url)

        if (tickets.length === 1) return new Response(null, { status: 503 })

        return Response.json({
          location: { directory: "/repo", project: { id: "project_1", directory: "/repo", canonical: "/repo" } },
          data: { ticket: "fixture-ticket", expires_in: 60 },
        })
      },
      { preconnect: fetch.preconnect },
    ),
  })

  // The production extension constructs its native PTY client internally. Replace only
  // the browser socket boundary; ticket acquisition still runs through the real client.
  const NativeWebSocket = globalThis.WebSocket

  // SAFETY: this isolated child substitutes only the browser socket constructor and
  // restores it below. Static ready-state constants retain their native values.
  globalThis.WebSocket = class {
    static OPEN = NativeWebSocket.OPEN
    static CLOSED = NativeWebSocket.CLOSED
    static CLOSING = NativeWebSocket.CLOSING
    static CONNECTING = NativeWebSocket.CONNECTING
    constructor(url: string | URL) {
      opened.push(url.toString())

      if (opened.length === 1) throw new Error("temporary connect failure")

      // SAFETY: the fixture socket implements the EventTarget, state, send and close
      // surface used by the real PTY client; no native network connection is opened.
      return socket as WebSocket
    }
  } as typeof WebSocket

  const server = {
    url: "https://server.example",
    client: {
      ...api,
      pty: {
        ...api.pty,
        get: async (input: { ptyID: string }) => {
          inspected.push(input.ptyID)

          return { data: { status: "running" } }
        },
        update: async () => undefined,
      },
    },
  }

  const host = document.createElement("div")
  document.body.append(host)
  const pty = { id: "pty_1", title: "Terminal 1", titleNumber: 1, cursor: 42 }
  const mod = await import("ghostty-web")
  const ghostty = await mod.Ghostty.load()

  const dispose = render(
    () =>
      createComponent(Terminal, {
        pty,
        server,
        directory: "/repo",
        ghostty: async () => ({ mod, ghostty }),
        onConnectError: (error: Error) => cloned.push(error),
      }),
    host,
  )

  try {
    for (let i = 0; i < 100 && tickets.length === 0; i++) await Bun.sleep(10)
    await Bun.sleep(30)
    expect(tickets.length).toBe(1)
    expect(opened.length).toBe(0)
    expect(cloned).toEqual([])

    for (let i = 0; i < 200 && opened.length < 2; i++) await Bun.sleep(10)
    expect(tickets.length).toBe(3)
    expect(opened.length).toBe(2)
    expect(inspected).toEqual(["pty_1", "pty_1"])
    expect(cloned).toEqual([])
    expect(opened.every((url) => url.includes("/pty/pty_1/connect?") && url.includes("cursor=42"))).toBe(true)
    expect(pty.id).toBe("pty_1")
    socket.dispatchEvent(new Event("open"))
  } finally {
    dispose()
    host.remove()
    globalThis.WebSocket = NativeWebSocket
  }
}, 10_000)
