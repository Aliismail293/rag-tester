# Hardware Regression Suite

The hardware regression suite referenced in the release process runs
212 automated test cases against a reference EL-7 unit before any
firmware release ships. The suite takes about 6 hours to run end to
end on a single rig, but Thistlewick Robotics runs it on 3 rigs in
parallel, bringing a full pass down to roughly 2 hours. Any single test
failure fails the whole suite; there is no mechanism to mark individual
tests as "known flaky" and skip them. The suite was last rewritten in
2023 to add torque-accuracy checks after the EL-7 launched.
