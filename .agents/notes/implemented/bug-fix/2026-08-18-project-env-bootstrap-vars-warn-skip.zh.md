# Agent Note: 项目 .env 中的启动专用变量改为告警而非崩溃

Status: implemented

[English](2026-08-18-project-env-bootstrap-vars-warn-skip.md) | 中文

## Problem

`loadLayeredEnv` 把调用目录的 `.env`(项目层)和 Harness home 的 `.env`(用户层)
作为凭据来源读取(见[用户环境分层决策](../architecture/2026-08-04-credentials-yaml-and-user-environment-layer.md)),
在应用任何值之前,对任意启动专用变量(`PATH`、`PYTHONPATH`、`NODE_OPTIONS`、
`DSH_*` 等)一律硬报错。但项目层是任意用户项目自己的文件:一个 Python 项目
为了自己的工具链合理地设置 `PYTHONPATH=./`,于是从这样的项目里启动 `dsh`
会以 `… .env sets "PYTHONPATH", which only the launching environment may set …`
中止——尽管 dsh 根本不需要这个变量。

## Decision

项目层遇到启动专用变量时改为**告警并丢弃**,不再抛错;用户层
(`$DSH_HOME/.env`,即 dsh 自己的配置)仍保留硬报错。两类文件里的启动专用值
**永远不会被应用**——只有继承自环境的变量才能提供它——所以"被发现的 `.env`
不能劫持进程、运行时、VCS 或网络引导"这一安全性质保持不变。项目层其余值照常应用。

`readEnvLayer` 增加 `reject` 参数(默认 `true`);`loadLayeredEnv` 对项目层传入
`false`。两层仍在任一值被应用之前先解析完毕,因此用户层一旦拒绝,项目层不会
被部分应用。

## Verification

`loadLayeredEnv` 单元测试:硬报错用例改为走用户层(`$DSH_HOME/.env`),仍断言抛错
且任何值都不被应用;新增用例断言一个含 `DSH_PERMISSION_MODE` 与 `PYTHONPATH`
的项目 `.env` 会告警、应用其余值、且从不落地这些启动专用变量。

## Alternatives considered

**从每个项目 `.env` 里删掉 `PYTHONPATH`。** dsh 的错误信息建议改为在 shell 里
export,但项目自己的 `.env` 为自己的工具链合理地携带这类变量;这只会变成逐个
项目打地鼠,下一个带启动专用变量的项目仍会崩溃。

**两层都告警+跳过。** 用户层是 dsh 自有的配置,那里的启动专用变量是真正的
dsh 配置错误,应当继续 fail-loud。

## Consequences

在任意项目里启动 dsh 不再因其项目的 `.env` 中止;那里的启动专用变量以单行告警
被忽略。`$DSH_HOME/.env` 的配置错误仍会 fail-loud。任何被发现的文件中的启动专用
变量始终不会被应用。
