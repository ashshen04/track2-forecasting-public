## Executive summary (read this first)

The first submission uses the tested `track2-first:20261007` Linux/amd64 image.
Push it to a public Docker Hub repository, verify an anonymous pull by digest,
then use the shared toolkit to create `submission.zip`. Upload the zip to the
Track 2 Development page on CodaBench. Keep the Team Key in the hidden terminal
prompt. This guide prepares an upload; it does not promise a forecast score.

## 1. 登录 Docker Hub

从仓库目录打开终端：

```bash
cd /Users/ash04/Code/track2-forecasting-public
docker login
```

打开终端提示的网页，输入设备码，用 `ashshen04` 登录。
密码只在 Docker 的登录页面输入。终端显示 `Login Succeeded` 后继续。

在 [Docker Hub](https://hub.docker.com/repositories/ashshen04) 点击
`Create repository`。选择 Namespace `ashshen04`，名称 `track2-first`，
Visibility `Public`。主办方必须能够在没有你的账号凭证时拉取镜像。

## 2. 上传程序镜像

镜像已经在本机构建。以下命令为它添加仓库名称并上传：

```bash
docker tag track2-first:20261007 ashshen04/track2-first:20261007
docker push ashshen04/track2-first:20261007
docker image inspect ashshen04/track2-first:20261007 --format '{{json .RepoDigests}}'
```

最后一条命令会显示形如 `ashshen04/track2-first@sha256:...` 的地址。
`sha256:...` 是不可变摘要，用来指定主办方实际运行的镜像内容。
使用上传后取得的摘要；不要用模型摘要或本地配置摘要代替。

如果需要重新构建，使用：

```bash
docker build --platform linux/amd64 -t track2-first:20261007 .
```

改动代码并重新上传后，要更新摘要并重新生成提交包。

## 3. 确认无需登录也能拉取

把下面的完整地址替换为上传后得到的实际摘要：

```bash
T2_IMAGE_REF='docker.io/ashshen04/track2-first@sha256:替换为实际64位摘要'
T2_DOCKER_HOST=$(docker context inspect --format '{{.Endpoints.docker.Host}}')
T2_ANON_CONFIG=$(mktemp -d /private/tmp/t2-anonymous.XXXXXX)
printf '{"auths":{"https://index.docker.io/v1/":{}}}\n' > "$T2_ANON_CONFIG/config.json"
docker --config "$T2_ANON_CONFIG" --host "$T2_DOCKER_HOST" \
  pull --platform linux/amd64 "$T2_IMAGE_REF"
```

临时配置仅有空的 Docker Hub 授权条目，不包含登录凭证。
显式指定 `linux/amd64`，因为这台 Mac 使用 arm64。
成功拉取才说明这个镜像允许匿名访问。

## 4. 生成提交文件

沿用上一步的 `T2_IMAGE_REF`。准备描述文件：

```bash
.venv/bin/python scripts/prepare_submission.py \
  --image "$T2_IMAGE_REF" \
  --out /private/tmp/t2-submission-01/submission.json
```

脚本从共享工具包的官方 Development 模板生成草稿，声明实际使用的 House
模型和 MIT 代码许可证。草稿故意不填写团队别名和描述摘要，这两项由正式
打包命令生成。不要直接上传草稿，也不要复制别人的 `team_id`。

把 `你的团队编号` 换成比赛网站显示的数字编号：

```bash
.venv/bin/qfbench2 submission pack \
  --descriptor /private/tmp/t2-submission-01/submission.json \
  --team-number 你的团队编号 \
  --out /private/tmp/t2-submission-01/submission.zip
```

终端会隐藏输入，让你输入比赛发放的 Team Key。它与 Docker 密码不同。
打包工具会自动生成团队别名和证明，zip 里只有 `submission.json` 和
`team-claim.json`。Team Key 不会放进 zip。

找不到团队编号或 Team Key 时，查看注册团队信息和主办方发给团队的通知。
仍找不到就通过比赛支持渠道询问，不要自己编造编号或密钥。

## 5. 在 CodaBench 上传

使用团队指定的 CodaBench 账号，打开主办方发给注册团队的 Track 2 比赛链接。
选择 Development 阶段，上传 `/private/tmp/t2-submission-01/submission.zip`。
上传的是 zip；镜像已在 Docker Hub，代码文件和镜像 tar 都不作为这个上传文件。

查看任务状态、评分和运行日志。尤其检查 `reasoning_applied` 和
`reasoning_skipped_reason`，确认正式环境里是否真的调用了 House 模型。
本地断网检查会回退到统计预测，因此不能证明真实 House 调用已经成功。

`/private/tmp` 可能被系统清理。完成后将正式 zip 复制到你自己的持久目录。
这份指南不会自动进行 CodaBench 上传，也不会消耗比赛提交次数。

## 官方规则

- [镜像公开拉取要求](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.4.4/docs/IMAGE-SUBMISSIONS.md)
- [描述文件与 zip 格式](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.4.4/starter-packs/track2/SUBMISSION-DESCRIPTOR.md)
- [House 模型身份](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md)
- [Docker 设备登录](https://docs.docker.com/reference/cli/docker/login/)
- [Docker Hub 创建仓库](https://docs.docker.com/docker-hub/repos/create/)
