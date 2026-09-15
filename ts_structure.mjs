import ts from "typescript";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const repo = path.resolve(process.argv[2] ?? "");
const project = process.argv[3];
if (!repo || !project) throw new Error("usage: node ts_structure.mjs REPO PROJECT");

const tracked = execFileSync(
  "git", ["-C", repo, "ls-files", "--cached", "--others", "--exclude-standard"],
  { encoding: "utf8" },
)
  .trim().split("\n").filter(Boolean)
  .filter((item) => /\.(?:tsx?|mts|cts|jsx?|mjs|cjs)$/.test(item));
const trackedSet = new Set(tracked);
const symbols = [];
const edges = [];

function lineOf(sourceFile, node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
}

function endLineOf(sourceFile, node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getEnd()).line + 1;
}

function declaredName(node) {
  if (!node?.name) return "";
  if (ts.isIdentifier(node.name) || ts.isPrivateIdentifier(node.name)) return node.name.text;
  return node.name.getText();
}

function namedFunction(node) {
  if (ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node)
    || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)
    || ts.isConstructorDeclaration(node)) {
    return declaredName(node) || (ts.isConstructorDeclaration(node) ? "constructor" : "");
  }
  if (ts.isFunctionExpression(node) && node.name) {
    return node.name.text;
  }
  if ((ts.isArrowFunction(node) || ts.isFunctionExpression(node)) && node.parent) {
    if (ts.isVariableDeclaration(node.parent) && ts.isIdentifier(node.parent.name)) {
      return node.parent.name.text;
    }
    if (ts.isPropertyAssignment(node.parent) || ts.isMethodDeclaration(node.parent)) {
      return declaredName(node.parent);
    }
  }
  return "";
}

function enclosingClassNames(node) {
  const result = [];
  for (let current = node.parent; current; current = current.parent) {
    if (ts.isClassDeclaration(current) && current.name) result.unshift(current.name.text);
  }
  return result;
}

function qualifiedName(node, name) {
  const parts = enclosingClassNames(node);
  for (let current = node.parent; current; current = current.parent) {
    if (ts.isFunctionLike(current)) {
      const parentName = namedFunction(current);
      if (parentName && parentName !== name && !parts.includes(parentName)) {
        parts.unshift(parentName);
      }
    }
  }
  parts.push(name);
  return parts.join(".");
}

function owner(node, file) {
  for (let current = node.parent; current; current = current.parent) {
    if (!ts.isFunctionLike(current)) continue;
    const name = namedFunction(current);
    if (name) return { name, qualified: qualifiedName(current, name) };
  }
  return { name: "<module>", qualified: file };
}

function addSymbol(sourceFile, file, node, kind, name = declaredName(node)) {
  if (!name) return;
  symbols.push({
    project,
    path: file,
    kind,
    name,
    qualified_name: qualifiedName(node, name),
    start_line: lineOf(sourceFile, node),
    end_line: endLineOf(sourceFile, node),
    signature: node.getText(sourceFile).split("{", 1)[0].slice(0, 500),
    metadata: {},
  });
}

function addEdge(
  sourceFile, file, node, relation, targetName, targetPath = null,
  metadata = {}, targetQualifiedName = null, confidence = "unresolved",
  sourceOverride = null,
) {
  if (!targetName) return;
  const source = sourceOverride ?? owner(node, file);
  edges.push({
    project,
    source_path: file,
    source_name: source.name,
    source_qualified_name: source.qualified,
    relation,
    target_name: targetName,
    target_path: targetPath,
    target_qualified_name: targetQualifiedName,
    line: lineOf(sourceFile, node),
    metadata,
    resolution_confidence: confidence,
  });
}

function resolveImport(file, specifier) {
  if (!specifier.startsWith(".") && !specifier.startsWith("@/")) return null;
  const raw = specifier.startsWith("@/")
    ? path.join("src", specifier.slice(2))
    : path.normalize(path.join(path.dirname(file), specifier));
  const extensions = [".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"];
  const candidates = [
    raw,
    ...extensions.map((extension) => raw + extension),
    ...extensions.map((extension) => path.join(raw, "index" + extension)),
  ];
  return candidates.find((candidate) => trackedSet.has(candidate)) ?? null;
}

function literalArgument(call, index = 0) {
  const argument = call.arguments[index];
  return argument && ts.isStringLiteralLike(argument) ? argument.text : "";
}

function compactExpression(node) {
  return node.getText().replace(/\s+/g, "").slice(0, 240);
}

function outerCallText(node) {
  let current = node;
  while (current.parent && (
    ts.isPropertyAccessExpression(current.parent)
    || ts.isElementAccessExpression(current.parent)
    || ts.isCallExpression(current.parent)
  )) current = current.parent;
  return compactExpression(current);
}

function callTarget(expression) {
  if (ts.isIdentifier(expression)) {
    return { root: expression.text, name: expression.text, expression: expression.text };
  }
  if (ts.isPropertyAccessExpression(expression)) {
    if (ts.isNewExpression(expression.expression)) {
      const constructorExpression = expression.expression.expression;
      if (ts.isPropertyAccessExpression(constructorExpression)
        && constructorExpression.expression.kind === ts.SyntaxKind.ThisKeyword
        && constructorExpression.name.text === "constructor") {
        return {
          root: "this.constructor",
          name: expression.name.text,
          expression: compactExpression(expression),
        };
      }
    }
    let root = expression.expression;
    while (ts.isPropertyAccessExpression(root)) root = root.expression;
    return {
      root: ts.isIdentifier(root) ? root.text : root.kind === ts.SyntaxKind.ThisKeyword ? "this" : "",
      name: expression.name.text,
      expression: compactExpression(expression),
    };
  }
  return { root: "", name: "", expression: compactExpression(expression) };
}

function resolveLocalTarget(localSymbols, target, node, file) {
  const candidates = localSymbols.get(target.name) ?? [];
  if (!candidates.length) return null;
  const classes = enclosingClassNames(node);
  const sameClass = candidates.find((symbol) =>
    classes.some((className) => symbol.qualified_name.startsWith(`${className}.`))
  );
  if (sameClass) return sameClass;
  return candidates.length === 1 ? candidates[0] : null;
}

function classBaseTarget(node, imports, localSymbols, file) {
  const clause = node.heritageClauses?.find(
    (item) => item.token === ts.SyntaxKind.ExtendsKeyword,
  );
  const expression = clause?.types?.[0]?.expression;
  if (!expression) return null;
  const target = callTarget(expression);
  if (!target.name) return null;
  const imported = imports.get(target.root);
  if (imported?.targetPath) {
    return {
      name: target.name,
      path: imported.targetPath,
      qualified: imported.namespace
        ? target.name
        : (imported.imported === "default" ? target.name : imported.imported),
      confidence: "import_resolved_inheritance",
    };
  }
  const local = (localSymbols.get(target.name) ?? []).find(
    (symbol) => symbol.kind === "class",
  );
  if (local) {
    return {
      name: target.name,
      path: file,
      qualified: local.qualified_name,
      confidence: "same_file_inheritance",
    };
  }
  return null;
}

for (const file of tracked) {
  const source = fs.readFileSync(path.join(repo, file), "utf8");
  const scriptKind = file.endsWith(".tsx") ? ts.ScriptKind.TSX
    : file.endsWith(".jsx") ? ts.ScriptKind.JSX
      : /\.(?:mjs|cjs|js)$/.test(file) ? ts.ScriptKind.JS : ts.ScriptKind.TS;
  const sourceFile = ts.createSourceFile(
    file, source, ts.ScriptTarget.Latest, true, scriptKind,
  );
  const imports = new Map();
  const localSymbols = new Map();
  const moduleName = path.basename(file, path.extname(file));
  symbols.push({
    project,
    path: file,
    kind: "module",
    name: moduleName,
    qualified_name: file,
    start_line: 1,
    end_line: sourceFile.getLineAndCharacterOfPosition(sourceFile.end).line + 1,
    signature: file,
    metadata: {},
  });

  function collectSymbols(node) {
    if (ts.isFunctionDeclaration(node)) addSymbol(sourceFile, file, node, "function");
    else if (ts.isFunctionExpression(node) && node.name
      && !ts.isVariableDeclaration(node.parent)
      && !ts.isPropertyAssignment(node.parent)) {
      addSymbol(sourceFile, file, node, "function", node.name.text);
    }
    else if (ts.isClassDeclaration(node)) addSymbol(sourceFile, file, node, "class");
    else if (ts.isInterfaceDeclaration(node)) addSymbol(sourceFile, file, node, "interface");
    else if (ts.isTypeAliasDeclaration(node)) addSymbol(sourceFile, file, node, "type");
    else if (ts.isEnumDeclaration(node)) addSymbol(sourceFile, file, node, "enum");
    else if (ts.isMethodDeclaration(node) || ts.isGetAccessorDeclaration(node)
      || ts.isSetAccessorDeclaration(node) || ts.isConstructorDeclaration(node)) {
      addSymbol(
        sourceFile, file, node, "method",
        declaredName(node) || (ts.isConstructorDeclaration(node) ? "constructor" : ""),
      );
    } else if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)
      && node.initializer
      && (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))) {
      const name = node.name.text;
      const kind = name.startsWith("use") ? "hook"
        : /^[A-Z]/.test(name) ? "component" : "function";
      addSymbol(sourceFile, file, node, kind, name);
    }
    ts.forEachChild(node, collectSymbols);
  }
  collectSymbols(sourceFile);
  for (const symbol of symbols) {
    if (symbol.path === file && symbol.kind !== "module") {
      const existing = localSymbols.get(symbol.name) ?? [];
      existing.push(symbol);
      localSymbols.set(symbol.name, existing);
    }
  }

  function collectEdges(node) {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      const specifier = node.moduleSpecifier.text;
      const targetPath = resolveImport(file, specifier);
      const clause = node.importClause;
      if (clause?.name) {
        imports.set(clause.name.text, {
          imported: "default", targetPath, namespace: false,
        });
      }
      if (clause?.namedBindings && ts.isNamedImports(clause.namedBindings)) {
        for (const element of clause.namedBindings.elements) {
          imports.set(element.name.text, {
            imported: element.propertyName?.text ?? element.name.text,
            targetPath,
            namespace: false,
          });
        }
      } else if (clause?.namedBindings && ts.isNamespaceImport(clause.namedBindings)) {
        imports.set(clause.namedBindings.name.text, {
          imported: "*", targetPath, namespace: true,
        });
      }
      addEdge(
        sourceFile, file, node, "imports", specifier, targetPath, {}, null,
        targetPath ? "exact_path" : "external_or_unresolved",
      );
    } else if (ts.isClassDeclaration(node) && node.name) {
      const className = node.name.text;
      const classQualifiedName = qualifiedName(node, className);
      const base = classBaseTarget(node, imports, localSymbols, file);
      if (base) {
        addEdge(
          sourceFile, file, node, "extends", base.name, base.path,
          { subclass: classQualifiedName }, base.qualified, base.confidence,
          { name: className, qualified: classQualifiedName },
        );
        for (const member of node.members) {
          if (!(ts.isMethodDeclaration(member)
            || ts.isGetAccessorDeclaration(member)
            || ts.isSetAccessorDeclaration(member))) continue;
          const methodName = declaredName(member);
          if (!methodName) continue;
          addEdge(
            sourceFile, file, member, "overrides", methodName, base.path,
            { base_class: base.name, subclass: classQualifiedName },
            `${base.qualified}.${methodName}`, base.confidence,
            { name: methodName, qualified: `${classQualifiedName}.${methodName}` },
          );
        }
      }
    } else if (ts.isCallExpression(node)) {
      const expression = compactExpression(node.expression);
      if (expression.endsWith(".rpc")) {
        const name = literalArgument(node);
        addEdge(
          sourceFile, file, node, "calls_rpc", name ? `public.${name}` : "",
          null, {}, name ? `public.${name}` : null, name ? "literal" : "unresolved",
        );
      } else if (expression.endsWith(".from")) {
        const name = literalArgument(node);
        const chain = outerCallText(node);
        const write = /\.(insert|update|upsert|delete)\(/.test(chain);
        addEdge(
          sourceFile, file, node, write ? "writes_table" : "reads_table",
          name ? `public.${name}` : "", null,
          { operation: write ? "write" : "read" },
          name ? `public.${name}` : null, name ? "literal" : "unresolved",
        );
      } else if (expression.endsWith(".functions.invoke")) {
        const name = literalArgument(node);
        addEdge(
          sourceFile, file, node, "invokes_edge_function", name, null, {}, name,
          name ? "literal" : "unresolved",
        );
      } else {
        const target = callTarget(node.expression);
        if (target.name) {
          const imported = imports.get(target.root);
          const local = (
            ts.isIdentifier(node.expression)
            || target.root === "this"
            || target.root === "this.constructor"
          ) ? resolveLocalTarget(localSymbols, target, node, file) : null;
          let targetPath = null;
          let targetQualifiedName = target.name;
          let confidence = "unresolved";
          if (imported) {
            targetPath = imported.targetPath;
            targetQualifiedName = imported.namespace
              ? target.name : (target.root === target.name ? imported.imported : target.name);
            confidence = targetPath ? "import_resolved" : "external_import";
          } else if (local) {
            targetPath = file;
            targetQualifiedName = local.qualified_name;
            confidence = "same_file";
          } else if (target.root === "this" || target.root === "this.constructor") {
            targetPath = file;
            confidence = "same_class_unresolved";
          }
          addEdge(
            sourceFile, file, node, "calls", target.name, targetPath,
            {
              expression: target.expression,
              receiver: target.root,
              argument_count: node.arguments.length,
              optional_chain: Boolean(node.questionDotToken),
              call_kind: ts.isIdentifier(node.expression) ? "identifier" : "property",
            }, targetQualifiedName, confidence,
          );
        }
      }
    }
    ts.forEachChild(node, collectEdges);
  }
  collectEdges(sourceFile);
}

const unresolvedEdges = edges.filter((edge) => edge.resolution_confidence === "unresolved").length;
const moduleOwnedEdges = edges.filter((edge) => edge.source_name === "<module>").length;
process.stdout.write(JSON.stringify({
  symbols,
  edges,
  coverage: {
    typescript_files: tracked.length,
    total_edges: edges.length,
    unresolved_edges: unresolvedEdges,
    module_owned_edges: moduleOwnedEdges,
    note: "Best-effort compiler AST graph; not a complete interprocedural analyzer.",
  },
}));
