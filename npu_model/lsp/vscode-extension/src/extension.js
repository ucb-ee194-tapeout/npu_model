"use strict";

const path = require("path");
const fs = require("fs");
const { workspace, window } = require("vscode");
const {
  LanguageClient,
  TransportKind,
} = require("vscode-languageclient/node");

let client;

function findProjectRoot(startDir) {
  let dir = startDir;
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(dir, "pyproject.toml"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  const folders = workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    return folders[0].uri.fsPath;
  }
  return startDir;
}

function buildServerOptions(serverScript, projectRoot, config) {
  const explicitPython = config.get("pythonPath", "").trim();
  const extraArgs = config.get("serverArgs", []);

  if (explicitPython) {
    return {
      command: explicitPython,
      args: [serverScript, ...extraArgs],
      options: { cwd: projectRoot },
      transport: TransportKind.stdio,
    };
  }

  return {
    command: "uv",
    args: [
      "run",
      "--with", "pygls",
      "--with", "lsprotocol",
      serverScript,
      ...extraArgs,
    ],
    options: { cwd: projectRoot },
    transport: TransportKind.stdio,
  };
}

function activate(context) {
  const config = workspace.getConfiguration("npu-asm");

  const realExtensionPath = fs.realpathSync(context.extensionPath);
  const serverScript = path.resolve(
    realExtensionPath,
    "..",
    "server.py"
  );

  context.subscriptions.push(
    require("vscode").commands.registerCommand("npu-asm.restartServer", async () => {
      if (!client) {
        window.showErrorMessage("NPU Assembly: language server is not running.");
        return;
      }
      await client.stop();
      client.start();
      window.showInformationMessage("NPU Assembly: language server restarted.");
    })
  );

  if (!fs.existsSync(serverScript)) {
    window.showErrorMessage(
      `NPU Assembly: cannot find LSP server at ${serverScript}`
    );
    return;
  }

  const projectRoot = findProjectRoot(
    path.resolve(realExtensionPath, "..", "..", "..")
  );

  const serverOptions = buildServerOptions(serverScript, projectRoot, config);

  const clientOptions = {
    documentSelector: [
      { scheme: "file", language: "npu-asm" },
    ],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher("**/*.{S,s,asm}"),
    },
    middleware: {},
  };

  client = new LanguageClient(
    "npu-asm",
    "NPU Assembly Language Server",
    serverOptions,
    clientOptions
  );

  client.start();
  context.subscriptions.push(client);
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };
