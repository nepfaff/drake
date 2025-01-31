# This file is automatically generated by upgrade.py.

ARCHIVES = [
    dict(
        name = "rust_darwin_aarch64__aarch64-apple-darwin__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_aarch64__aarch64-apple-darwin__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_darwin_aarch64__aarch64-apple-darwin__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_aarch64__aarch64-apple-darwin__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "c16ba325a1c491d9963b0095cbae85e1a7d51e1bcf10426891e297a190ddb236",
                    "stripPrefix": "rustc-1.75.0-aarch64-apple-darwin/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "5a862003b2873632a5154d6ea3174ddcf5b39de8792c1970f3f8f65c23f0108b",
                    "stripPrefix": "clippy-1.75.0-aarch64-apple-darwin/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "86320d22c192b7a531daed941bbc11d8c7d01b6490cb4b85e7aa7ff92b5baf65",
                    "stripPrefix": "cargo-1.75.0-aarch64-apple-darwin/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "27627c5606145a3a6cf2d12a75e8f3ee4b8be08dc5485d7c5a0e7c7c6245bfb9",
                    "stripPrefix": "rustfmt-1.75.0-aarch64-apple-darwin/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "b9001b9a746cb88cdbf011f8622df81905131aa6ba55fb6bbe1eacc895da488c",
                    "stripPrefix": "llvm-tools-1.75.0-aarch64-apple-darwin/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "25e9849c4bd0032599e31a1358c7d175cfded3807593f6f1b5d9742db4941355",
                    "stripPrefix": "rust-std-1.75.0-aarch64-apple-darwin/rust-std-aarch64-apple-darwin",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-aarch64-apple-darwin.tar.xz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_darwin_x86_64__x86_64-apple-darwin__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_x86_64__x86_64-apple-darwin__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_darwin_x86_64__x86_64-apple-darwin__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_x86_64__x86_64-apple-darwin__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "5c2f0c923933682611bc683c4abe7558682b4a9e3b716e28ee24d8def0a888f1",
                    "stripPrefix": "rustc-1.75.0-x86_64-apple-darwin/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "5f7b7116430f7816b08121e84daacb0e9e0bca35470fc858a3b731149b97dddd",
                    "stripPrefix": "clippy-1.75.0-x86_64-apple-darwin/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "08c594b582141bfb3113b4325f567abe1cae5d5e075b0b2b56553f8bc59486b5",
                    "stripPrefix": "cargo-1.75.0-x86_64-apple-darwin/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "5604910e80182ded082403485953ab88c5b2eb9e0d2c7d3e860ed96dca32d2f2",
                    "stripPrefix": "rustfmt-1.75.0-x86_64-apple-darwin/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "eed6cb1ee9bc1a7324fee6b5c3104f828844ad0529cf43cc6c5db5af1d7ca06f",
                    "stripPrefix": "llvm-tools-1.75.0-x86_64-apple-darwin/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
                {
                    "sha256": "a1b2fee1b1b04b15a40ce9dd1e395502f157949a07e9edba7a015b03b35b77b2",
                    "stripPrefix": "rust-std-1.75.0-x86_64-apple-darwin/rust-std-x86_64-apple-darwin",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-x86_64-apple-darwin.tar.xz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_linux_aarch64__aarch64-unknown-linux-gnu__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_aarch64__aarch64-unknown-linux-gnu__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_linux_aarch64__aarch64-unknown-linux-gnu__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_aarch64__aarch64-unknown-linux-gnu__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "b1d7bb8b0420b71585cf9c4eb5fd1e986fd83edc2d393510b54a9b20272386a3",
                    "stripPrefix": "rustc-1.75.0-aarch64-unknown-linux-gnu/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "de0335751e3992e1ac3425e890669c8ebd51c89de9126ada93ba8dc0045a8b84",
                    "stripPrefix": "clippy-1.75.0-aarch64-unknown-linux-gnu/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "cf367bccbc97ba86b4cf8a0141c9c270523e38f865dc7220b3cfdd79b67200ed",
                    "stripPrefix": "cargo-1.75.0-aarch64-unknown-linux-gnu/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "5869ca3203e6f3ade284385eed00f07847b287188da8693e7948ddfd05974d8b",
                    "stripPrefix": "rustfmt-1.75.0-aarch64-unknown-linux-gnu/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "4941b158c0bca9a7e58df47909d146d71e34d192b5cfc553bca855333d5a74d3",
                    "stripPrefix": "llvm-tools-1.75.0-aarch64-unknown-linux-gnu/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "2ea0dc380ac1fced245bafadafd0da50167a4a416b6011e3d73ba3e657a71d15",
                    "stripPrefix": "rust-std-1.75.0-aarch64-unknown-linux-gnu/rust-std-aarch64-unknown-linux-gnu",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-aarch64-unknown-linux-gnu.tar.xz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_linux_x86_64__x86_64-unknown-linux-gnu__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_x86_64__x86_64-unknown-linux-gnu__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_linux_x86_64__x86_64-unknown-linux-gnu__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_x86_64__x86_64-unknown-linux-gnu__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "2824ba4045acdddfa436da4f0bb72807b64a089aa2e7c9a66ca1a3a571114ce7",
                    "stripPrefix": "rustc-1.75.0-x86_64-unknown-linux-gnu/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "860fc6dc0df2d7e27886c57af38a86020ea1fe4878b9386b9f762f86b59776ca",
                    "stripPrefix": "clippy-1.75.0-x86_64-unknown-linux-gnu/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "6ac164e7da969a1d524f747f22792e9aa08bc7446f058314445a4f3c1d31a6bd",
                    "stripPrefix": "cargo-1.75.0-x86_64-unknown-linux-gnu/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "1799f103dfaa19b8cf4781500773c709c627c00d203aecd1b56d75d8e624e4d5",
                    "stripPrefix": "rustfmt-1.75.0-x86_64-unknown-linux-gnu/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "e429421b2d081c668ffcfbf8cc673739371adfff1b4234d92130ccc088ba82c2",
                    "stripPrefix": "llvm-tools-1.75.0-x86_64-unknown-linux-gnu/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
                {
                    "sha256": "136b132199f7bbda2aa0bbff6d1e6ae7d5fca2994a2f2a432a5e99de224b6314",
                    "stripPrefix": "rust-std-1.75.0-x86_64-unknown-linux-gnu/rust-std-x86_64-unknown-linux-gnu",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-x86_64-unknown-linux-gnu.tar.xz",
                    ],
                },
            ],
        ),
    ),
]
