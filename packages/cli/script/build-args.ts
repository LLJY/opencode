export function validateBuildArgs(args: string[], options: { flags: string[]; values: string[] }) {
  const seen = new Set<string>()
  for (const arg of args) {
    const name = arg.split("=", 1)[0]
    if (!options.flags.includes(arg) && !options.values.includes(name)) throw new Error(`Unknown build flag: ${arg}`)
    if (seen.has(name)) throw new Error(`Duplicate build flag: ${name}`)
    seen.add(name)
    if (options.values.includes(name) && (!arg.startsWith(`${name}=`) || !arg.slice(name.length + 1).trim())) {
      throw new Error(`${name} requires a value`)
    }
  }
}
